#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量评估 GMR / 网络 retargeting 误差的脚本（mpbpe）：

单个 pkl 的评估逻辑与原版一致（兼容原脚本输出）。
- 从 pkl 中读取 robot_motion（qpos）和 human_motion（IK target）
- 用 MuJoCo 对机器人做 FK，得到各个 body 的世界坐标
- 从 human_motion 里还原各个人体 anchor 的世界坐标
- 根据 bvh_to_g1.json 里的 human↔robot 对应关系，对应配对后计算：
    - Eᵍ-mpbpe（global）
    - E_mpbpe（root-relative）
并支持：
- 对一个文件夹内的所有 *.pkl（可递归）逐个评估
- 结果可选写出 CSV/JSON 汇总
"""

import argparse
import csv
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import mujoco as mj


# ----------------------------
# 工具函数：从 pkl 还原机器人 qpos
# ----------------------------

def load_robot_qpos_from_motion_dict(motion: Dict) -> np.ndarray:
    """
    motion: 形如 bvh_to_robot.py 里保存的 robot_motion：
        {
            "fps": ...,
            "root_pos": (T, 3),
            "root_rot": (T, 4),  # 保存时是 xyzw
            "dof_pos":  (T, ndof),
            ...
        }
    返回：
        qpos: (T, nq) = [root_pos(3), root_quat(wxyz 4), dof_pos(...)]
    """
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)  # (T, 3)
    root_rot_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)  # (T, 4)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)  # (T, ndof)

    T = root_pos.shape[0]
    if root_rot_xyzw.shape[0] != T or dof_pos.shape[0] != T:
        raise ValueError(f"robot_motion 时间维不一致: root_pos={root_pos.shape}, root_rot={root_rot_xyzw.shape}, dof_pos={dof_pos.shape}")

    nq = 3 + 4 + dof_pos.shape[1]
    qpos = np.zeros((T, nq), dtype=np.float64)

    # 平移
    qpos[:, :3] = root_pos

    # 注意：保存时是 xyzw，MuJoCo qpos 里要求 wxyz
    qpos[:, 3:7] = root_rot_xyzw[:, [3, 0, 1, 2]]

    # 关节角
    qpos[:, 7:] = dof_pos

    return qpos


# ----------------------------
# FK：机器人 body 世界坐标
# ----------------------------

def get_body_ids(model: mj.MjModel, robot_body_names: Sequence[str]) -> List[int]:
    body_ids = []
    for name in robot_body_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in model.")
        body_ids.append(bid)
    return body_ids


def fk_robot_bodies_with_ids(model: mj.MjModel, qpos_seq: np.ndarray, body_ids: Sequence[int]) -> np.ndarray:
    """
    输入：
        qpos_seq: (T, nq)
        body_ids: len=J 的 body id 列表
    返回：
        xpos: (T, J, 3)
    """
    data = mj.MjData(model)
    T = int(qpos_seq.shape[0])
    J = len(body_ids)

    xpos = np.zeros((T, J, 3), dtype=np.float64)
    for t in range(T):
        data.qpos[:] = qpos_seq[t]
        mj.mj_forward(model, data)
        for j, bid in enumerate(body_ids):
            xpos[t, j] = data.xpos[bid]
    return xpos


# ----------------------------
# 从 human_motion 还原 human anchor 世界坐标
# ----------------------------

def load_human_global_pos_from_motion_dict(human_motion: Dict) -> Tuple[np.ndarray, List[str]]:
    """
    human_motion 结构（参照你现在 bvh_to_robot.py 的保存方式）：
        {
            "fps": ...,
            "root_pos": (T, 3),           # Hips anchor 的 global pos
            "root_rot": (T, 4),           # root rot (xyzw)
            "dof_pos": None,
            "local_body_pos": (T, N, 3),  # 每个关节相对 root 的位移
            "link_body_list": [N 个 human 关节名字],
        }
    返回：
        human_global_pos: (T, N, 3)  # 世界坐标下所有关节 anchor 的位置
        human_body_names: list[str]
    """
    root_pos = np.asarray(human_motion["root_pos"], dtype=np.float64)           # (T, 3)
    local_body_pos = np.asarray(human_motion["local_body_pos"], dtype=np.float64)  # (T, N, 3)

    T, N = local_body_pos.shape[:2]
    if root_pos.shape[0] != T:
        raise ValueError(f"human_motion 时间维不一致: root_pos={root_pos.shape}, local_body_pos={local_body_pos.shape}")

    human_global_pos = local_body_pos + root_pos[:, None, :]  # (T, N, 3)
    human_body_names = list(human_motion["link_body_list"])
    if len(human_body_names) != N:
        raise ValueError(f"human_motion['link_body_list'] 长度与 local_body_pos 的 N 不一致: {len(human_body_names)} vs {N}")

    return human_global_pos, human_body_names


# ----------------------------
# ik_match_table 解析
# ----------------------------

def build_pairs_from_ik_cfg(ik_cfg: Dict, tables: str) -> List[Tuple[str, str, float]]:
    """
    返回 pairs: (robot_body_name, human_body_name, pos_weight)
    """
    use_table1 = tables in ("1", "both") and ik_cfg.get("use_ik_match_table1", True)
    use_table2 = tables in ("2", "both") and ik_cfg.get("use_ik_match_table2", True)

    pairs: List[Tuple[str, str, float]] = []
    seen = set()

    if use_table1:
        for robot_body_name, entry in ik_cfg["ik_match_table1"].items():
            human_name, pos_w, rot_w, pos_off, rot_off = entry
            if pos_w >= 0:
                key = (robot_body_name, human_name)
                if key not in seen:
                    pairs.append((robot_body_name, human_name, float(pos_w)))
                    seen.add(key)

    if use_table2:
        for robot_body_name, entry in ik_cfg["ik_match_table2"].items():
            human_name, pos_w, rot_w, pos_off, rot_off = entry
            if pos_w > 0:
                key = (robot_body_name, human_name)
                if key not in seen:
                    pairs.append((robot_body_name, human_name, float(pos_w)))
                    seen.add(key)

    if not pairs:
        raise ValueError("在 ik_match_table 中没有找到 pos_weight > 0 的关节，无法计算 mpbpe。")

    return pairs


def find_root_index(pairs: Sequence[Tuple[str, str, float]], ik_cfg: Dict) -> int:
    robot_root_name = ik_cfg.get("robot_root_name", None)
    human_root_name = ik_cfg.get("human_root_name", None)
    if robot_root_name is None or human_root_name is None:
        return 0

    for j, (rb, hn, _) in enumerate(pairs):
        if rb == robot_root_name and hn == human_root_name:
            return j
    return 0


# ----------------------------
# 误差统计
# ----------------------------

def compute_stats_mm(dist_flat_m: np.ndarray) -> Dict[str, float]:
    """
    dist_flat_m: (K,) 单位 m
    返回：min/median/mean/max（单位 mm）
    """
    dist_flat_m = np.asarray(dist_flat_m, dtype=np.float64)
    if dist_flat_m.size == 0:
        raise ValueError("dist_flat 为空，无法统计。")

    mn = float(dist_flat_m.min()) * 1000.0
    md = float(np.median(dist_flat_m)) * 1000.0
    me = float(dist_flat_m.mean()) * 1000.0
    mx = float(dist_flat_m.max()) * 1000.0
    return {"min_mm": mn, "median_mm": md, "mean_mm": me, "max_mm": mx}


@dataclass
class EvalResult:
    file: str
    status: str  # "ok" or "error"
    error: str = ""

    T: int = 0
    J: int = 0

    Eg_min_mm: float = np.nan
    Eg_med_mm: float = np.nan
    Eg_mean_mm: float = np.nan
    Eg_max_mm: float = np.nan

    E_min_mm: float = np.nan
    E_med_mm: float = np.nan
    E_mean_mm: float = np.nan
    E_max_mm: float = np.nan


def evaluate_one_pkl(
    pkl_path: Path,
    model: mj.MjModel,
    body_ids: Sequence[int],
    pairs: Sequence[Tuple[str, str, float]],
    root_index: int,
    verbose: bool = False,
) -> EvalResult:
    """
    对单个 pkl 进行评估。
    """
    try:
        with pkl_path.open("rb") as f:
            data = pickle.load(f)

        robot_motion = data["robot_motion"]
        human_motion = data["human_motion"]
        if human_motion is None:
            raise ValueError("pkl 中 human_motion 为 None，请确认保存逻辑。")

        qpos_robot = load_robot_qpos_from_motion_dict(robot_motion)  # (T, nq)

        human_global_pos, human_body_names = load_human_global_pos_from_motion_dict(human_motion)  # (T, Nh, 3)
        human_name_to_idx = {name: i for i, name in enumerate(human_body_names)}

        # 根据 pairs 构造 human_indices（对每个 pkl 都重新匹配一次，避免名字/顺序差异）
        human_indices = []
        for _, human_name, _ in pairs:
            if human_name not in human_name_to_idx:
                raise ValueError(f"Human 关节 {human_name} 不在 human_motion['link_body_list'] 中。")
            human_indices.append(human_name_to_idx[human_name])

        if verbose:
            print(f"\n[{pkl_path.name}] 评估关节对（robot ↔ human）：")
            for j, (rb, hn, _) in enumerate(pairs):
                print(f"  {j}: {rb}  <->  {hn}")

        # 时间对齐：截到最短
        T_robot = int(qpos_robot.shape[0])
        T_h = int(human_global_pos.shape[0])
        T = min(T_robot, T_h)
        if T <= 0:
            raise ValueError(f"序列长度异常: T_robot={T_robot}, T_human={T_h}")

        qpos_robot = qpos_robot[:T]
        human_global_pos = human_global_pos[:T]

        # 机器人 FK
        xpos_robot = fk_robot_bodies_with_ids(model, qpos_robot, body_ids)  # (T, J, 3)

        # 人体对应 joints
        xpos_human = np.take(human_global_pos, indices=human_indices, axis=1)  # (T, J, 3)

        # Global
        dist_global = np.linalg.norm(xpos_human - xpos_robot, axis=-1)  # (T, J)
        g = compute_stats_mm(dist_global.reshape(-1))

        # Root-relative
        ref_root = xpos_human[:, root_index:root_index + 1, :]
        pred_root = xpos_robot[:, root_index:root_index + 1, :]
        dist_rel = np.linalg.norm((xpos_human - ref_root) - (xpos_robot - pred_root), axis=-1)
        r = compute_stats_mm(dist_rel.reshape(-1))

        J = int(len(pairs))
        return EvalResult(
            file=str(pkl_path),
            status="ok",
            T=T,
            J=J,
            Eg_min_mm=g["min_mm"], Eg_med_mm=g["median_mm"], Eg_mean_mm=g["mean_mm"], Eg_max_mm=g["max_mm"],
            E_min_mm=r["min_mm"],  E_med_mm=r["median_mm"],  E_mean_mm=r["mean_mm"],  E_max_mm=r["max_mm"],
        )

    except Exception as e:
        return EvalResult(file=str(pkl_path), status="error", error=str(e))


# ----------------------------
# 批量遍历
# ----------------------------

def collect_pkls(pkl_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    if recursive:
        files = sorted(pkl_dir.rglob(pattern))
    else:
        files = sorted(pkl_dir.glob(pattern))
    files = [p for p in files if p.is_file()]
    return files


def write_csv(results: List[EvalResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file", "status", "error",
        "T", "J",
        "Eg_min_mm", "Eg_med_mm", "Eg_mean_mm", "Eg_max_mm",
        "E_min_mm", "E_med_mm", "E_mean_mm", "E_max_mm",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in fieldnames})


def write_json(results: List[EvalResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.__dict__ for r in results]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(results: List[EvalResult]) -> None:
    ok = [r for r in results if r.status == "ok"]
    bad = [r for r in results if r.status != "ok"]

    print("\n================ Batch Summary ================")
    print(f"Total: {len(results)} | OK: {len(ok)} | Error: {len(bad)}")

    if ok:
        Eg_means = np.array([r.Eg_mean_mm for r in ok], dtype=np.float64)
        E_means = np.array([r.E_mean_mm for r in ok], dtype=np.float64)
        print("----------------------------------------------")
        print("Eᵍ-mpbpe mean(mm) across files:")
        print(f"  mean={Eg_means.mean():.3f} | median={np.median(Eg_means):.3f} | min={Eg_means.min():.3f} | max={Eg_means.max():.3f}")
        print("E_mpbpe mean(mm) across files:")
        print(f"  mean={E_means.mean():.3f} | median={np.median(E_means):.3f} | min={E_means.min():.3f} | max={E_means.max():.3f}")

    if bad:
        print("----------------------------------------------")
        print("Errors (showing up to 10):")
        for r in bad[:10]:
            print(f"  - {Path(r.file).name}: {r.error}")
    print("==============================================\n")


# ----------------------------
# main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    # 单文件模式（兼容原脚本）
    parser.add_argument("--pkl", type=str, default=None,
                        help="单个 pkl 路径（兼容原脚本）。若指定 --pkl_dir，则忽略此项。")

    # 批量模式
    parser.add_argument("--pkl_dir", type=str, default=None,
                        help="包含多个 pkl 的目录。指定后进入批量评估模式。")
    parser.add_argument("--glob", type=str, default="*.pkl",
                        help="批量模式下的匹配规则，默认 *.pkl")
    parser.add_argument("--recursive", action="store_true",
                        help="批量模式下递归搜索子目录")

    # 公共参数
    parser.add_argument("--robot_xml", type=str, required=True,
                        help="MuJoCo 机器人 xml 路径（例如 g1.xml）")
    parser.add_argument("--ik_config", type=str, required=True,
                        help="IK 配置 json（例如 bvh_to_g1.json）")
    parser.add_argument("--tables", type=str, default="both",
                        choices=["1", "2", "both"],
                        help="使用 ik_match_table1 / 2 / both 中的哪些关节（默认 both）")

    # 输出
    parser.add_argument("--out_csv", type=str, default=None, help="可选：写出 CSV 汇总结果")
    parser.add_argument("--out_json", type=str, default=None, help="可选：写出 JSON 汇总结果")

    # 日志
    parser.add_argument("--verbose", action="store_true", help="打印关节对映射（每个文件都会打印）")
    parser.add_argument("--print_each", action="store_true", help="逐文件打印一行结果（推荐批量时打开）")
    parser.add_argument("--fail_fast", action="store_true", help="遇到错误立刻退出（默认不中断）")

    args = parser.parse_args()

    # 1) 读取 ik_config
    with Path(args.ik_config).open("r", encoding="utf-8") as f:
        ik_cfg = json.load(f)

    pairs = build_pairs_from_ik_cfg(ik_cfg, tables=args.tables)
    robot_body_names = [rb for rb, _, _ in pairs]
    root_index = find_root_index(pairs, ik_cfg)

    # 2) MuJoCo 模型（只加载一次）
    model = mj.MjModel.from_xml_path(args.robot_xml)
    body_ids = get_body_ids(model, robot_body_names)

    # 3) 单文件 or 批量
    results: List[EvalResult] = []

    if args.pkl_dir is not None:
        pkl_dir = Path(args.pkl_dir)
        if not pkl_dir.exists():
            raise FileNotFoundError(f"--pkl_dir not found: {pkl_dir}")

        files = collect_pkls(pkl_dir, pattern=args.glob, recursive=args.recursive)
        if not files:
            raise FileNotFoundError(f"No pkl matched in {pkl_dir} with pattern='{args.glob}' (recursive={args.recursive})")

        print(f"Found {len(files)} pkl files in {pkl_dir} (pattern='{args.glob}', recursive={args.recursive})")
        # 批量评估
        for i, pkl_path in enumerate(files, 1):
            r = evaluate_one_pkl(
                pkl_path=pkl_path,
                model=model,
                body_ids=body_ids,
                pairs=pairs,
                root_index=root_index,
                verbose=args.verbose,
            )
            results.append(r)

            if args.print_each:
                if r.status == "ok":
                    print(f"[{i:4d}/{len(files):4d}] {pkl_path.name} | "
                          f"Eg_mean={r.Eg_mean_mm:.3f}mm, E_mean={r.E_mean_mm:.3f}mm | T={r.T}, J={r.J}")
                else:
                    print(f"[{i:4d}/{len(files):4d}] {pkl_path.name} | ERROR: {r.error}")

            if args.fail_fast and r.status != "ok":
                break

    else:
        if args.pkl is None:
            raise ValueError("请指定 --pkl 或 --pkl_dir 之一。")

        pkl_path = Path(args.pkl)
        r = evaluate_one_pkl(
            pkl_path=pkl_path,
            model=model,
            body_ids=body_ids,
            pairs=pairs,
            root_index=root_index,
            verbose=True,  # 单文件默认打印映射，方便检查
        )
        results = [r]

        # 单文件模式保持原脚本那种“块状输出”
        if r.status == "ok":
            print("======================================")
            print(f"pkl: {pkl_path}")
            print(f"使用 ik_match_table: {args.tables}")
            print("--------------------------------------")
            print("Eᵍ-mpbpe (Global, 单位 mm):")
            print(f"  min   = {r.Eg_min_mm:.3f}")
            print(f"  median= {r.Eg_med_mm:.3f}")
            print(f"  mean  = {r.Eg_mean_mm:.3f}")
            print(f"  max   = {r.Eg_max_mm:.3f}")
            print("--------------------------------------")
            print("E_mpbpe (Root-relative, 单位 mm):")
            print(f"  min   = {r.E_min_mm:.3f}")
            print(f"  median= {r.E_med_mm:.3f}")
            print(f"  mean  = {r.E_mean_mm:.3f}")
            print(f"  max   = {r.E_max_mm:.3f}")
            print("======================================")
        else:
            print(f"[ERROR] {r.error}")

    # 4) 汇总
    if len(results) > 1:
        print_summary(results)

    # 5) 输出文件
    if args.out_csv:
        write_csv(results, Path(args.out_csv))
        print(f"[Saved] CSV -> {args.out_csv}")
    if args.out_json:
        write_json(results, Path(args.out_json))
        print(f"[Saved] JSON -> {args.out_json}")


if __name__ == "__main__":
    main()

"""
示例：

# 单文件（兼容原脚本）
python tools/eval_mpbpe_batch.py \
  --pkl retargeting_result/net_out_pkls_refine/aiming2_subject5_refine.pkl \
  --robot_xml assets/unitree_g1/g1_mocap_29dof.xml \
  --ik_config general_motion_retargeting/ik_configs/bvh_to_g1.json \
  --tables both

# 批量：评估某个文件夹下所有 pkl（不递归）
python tools/eval_mpbpe_batch.py \
  --pkl_dir retargeting_result/net_loss_jerk_target/net_out_refine\
  --glob "*.pkl" \
  --robot_xml assets/unitree_g1/g1_mocap_29dof.xml \
  --ik_config general_motion_retargeting/ik_configs/bvh_to_g1.json \
  --tables both \
  --print_each \
  --out_csv retargeting_result/net_loss_jerk_target/net_out_refine/eval_mpbpe_refine.csv

# 批量：递归子目录
python tools/eval_mpbpe_batch.py \
  --pkl_dir retargeting_result \
  --glob "*.pkl" \
  --recursive \
  --robot_xml assets/unitree_g1/g1_mocap_29dof.xml \
  --ik_config general_motion_retargeting/ik_configs/bvh_to_g1.json \
  --tables both \
  --print_each
"""