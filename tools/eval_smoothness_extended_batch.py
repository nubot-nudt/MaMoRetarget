#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扩展版平滑度评估脚本（支持批量评估文件夹）：同时评估
  1) 机器人 / 人体 anchor 位置轨迹的 SPARC
  2) 机器人各关节角时间序列的平滑度：关节角 SPARC + 关节角 jerk RMS

相较原版：原脚本只支持单个 --pkl 文件；本版新增 --pkl_dir / --pattern / --recursive
并支持把每个文件的汇总结果输出到 CSV/JSON。

用法示例：

(1) 单文件（默认 detailed，与原版类似的逐关节/逐DOF打印）
    python tools/eval_smoothness_extended.py \
        --pkl retargeting_result/lafan1_g1_save/aiming1_subject4_refine.pkl \
        --robot_xml assets/unitree_g1/g1_mocap_29dof.xml \
        --ik_config general_motion_retargeting/ik_configs/bvh_to_g1.json \
        --tables both

(2) 文件夹批量（默认 brief，仅每个文件打印一行汇总；加 --detailed 可展开）
    python tools/eval_smoothness_extended.py \
        --pkl_dir retargeting_result/lafan1_g1_save \
        --pattern "*.pkl" \
        --robot_xml assets/unitree_g1/g1_mocap_29dof.xml \
        --ik_config general_motion_retargeting/ik_configs/bvh_to_g1.json \
        --tables both \
        --out_csv smoothness_summary.csv

注意：本脚本不会改动任何数据，只做评估、打印/导出。
"""

import argparse
import csv
import json
import pickle
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import mujoco as mj


# ===================== SPARC 实现 =====================

def _sparc_1d_speed(speed, fs, padlevel=4, fc_max=10.0, amp_th=0.05):
    """
    对 1D 速度标量序列计算 SPARC。

    speed: (T,) 速度标量（非负），采样频率 fs (Hz)
    padlevel: 频谱插值倍数（通过零填充提高频率分辨率）
    fc_max: 最高考虑频率（Hz），一般 10~20
    amp_th: 归一化谱幅阈值

    返回：sparc (float). 注意通常为负数；越接近 0 越平滑。
    """
    speed = np.asarray(speed, dtype=float)

    # 全零或常数轨迹没有实际意义
    if speed.size == 0 or np.allclose(speed, speed[0]):
        return np.nan

    T = len(speed)
    n_fft = int(2 ** np.ceil(np.log2(T * padlevel)))

    v_fft = np.fft.rfft(speed, n=n_fft)
    mag = np.abs(v_fft)

    if mag[0] <= 0:
        return np.nan

    vhat = mag / mag[0]
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    omegas = 2.0 * np.pi * freqs

    idx_fc_max = np.searchsorted(freqs, fc_max, side="right") - 1
    idx_fc_max = max(1, idx_fc_max)

    above_th = np.where(vhat >= amp_th)[0]
    idx_thr = int(above_th[-1]) if len(above_th) > 0 else 1

    idx_c = int(min(idx_fc_max, idx_thr))
    idx_c = max(1, idx_c)

    omega_c = omegas[idx_c]
    x = omegas[:idx_c + 1] / omega_c
    y = vhat[:idx_c + 1]

    dx = np.diff(x)
    dy = np.diff(y)

    L = np.sum(np.sqrt(dx ** 2 + dy ** 2))
    return -float(L)


def sparc_from_3d_position(pos, fs, padlevel=4, fc_max=10.0, amp_th=0.05):
    """
    对 3D 位置轨迹 pos(t,3) 计算 SPARC：
      1) 计算速度向量 v(t)
      2) 取速度模长 ||v(t)|| 作为 1D 速度标量
      3) 调用 _sparc_1d_speed
    """
    pos = np.asarray(pos, dtype=float)
    if not (pos.ndim == 2 and pos.shape[1] == 3):
        raise ValueError("pos 必须是 (T,3)")

    dt = 1.0 / fs
    vel = np.gradient(pos, dt, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    return _sparc_1d_speed(speed, fs, padlevel=padlevel, fc_max=fc_max, amp_th=amp_th)


def sparc_from_1d_signal(sig, fs, padlevel=4, fc_max=10.0, amp_th=0.05):
    """
    对 1D 信号 q(t) 计算“角度版 SPARC”:
      1) 计算 dq/dt
      2) 取 |dq/dt| 作为速度标量
      3) 调用 _sparc_1d_speed
    """
    sig = np.asarray(sig, dtype=float)
    if sig.ndim != 1:
        raise ValueError("sparc_from_1d_signal 期望 1D 序列")

    dt = 1.0 / fs
    vel = np.gradient(sig, dt)
    speed = np.abs(vel)
    return _sparc_1d_speed(speed, fs, padlevel=padlevel, fc_max=fc_max, amp_th=amp_th)


def jerk_rms_1d(sig, fs):
    """
    jerk RMS: j = d^3 q / dt^3 的 RMS
    """
    sig = np.asarray(sig, dtype=float)
    if sig.ndim != 1:
        raise ValueError("jerk_rms_1d 期望 1D 序列")

    dt = 1.0 / fs
    v = np.gradient(sig, dt)
    a = np.gradient(v, dt)
    j = np.gradient(a, dt)
    return float(np.sqrt(np.mean(j ** 2)))


# ===================== 数据相关工具 =====================

def build_robot_qpos(robot_motion):
    """
    由 robot_motion 还原 qpos 序列 (T, nq)：
      - root_pos: (T,3)
      - root_rot: (T,4)  存的是 xyzw
      - dof_pos : (T, ndof)
    -> qpos = [root_pos, root_quat(wxyz), dof_pos]
    """
    root_pos = np.asarray(robot_motion["root_pos"], dtype=float)
    root_rot_xyzw = np.asarray(robot_motion["root_rot"], dtype=float)
    dof_pos = np.asarray(robot_motion["dof_pos"], dtype=float)

    T = root_pos.shape[0]
    if root_rot_xyzw.shape[0] != T or dof_pos.shape[0] != T:
        raise ValueError("root_pos/root_rot/dof_pos 的时间长度不一致")

    nq = 3 + 4 + dof_pos.shape[1]
    qpos = np.zeros((T, nq), dtype=float)

    qpos[:, :3] = root_pos
    # xyzw -> wxyz
    qpos[:, 3:7] = root_rot_xyzw[:, [3, 0, 1, 2]]
    qpos[:, 7:] = dof_pos
    return qpos


def fk_robot_bodies_by_id(model: mj.MjModel, qpos_seq: np.ndarray, body_ids: List[int]) -> np.ndarray:
    """
    用 MuJoCo FK 求出给定 body_ids 的世界坐标。
    返回：xpos: (T, J, 3)
    """
    data = mj.MjData(model)
    T = qpos_seq.shape[0]
    J = len(body_ids)
    xpos = np.zeros((T, J, 3), dtype=float)

    for t in range(T):
        data.qpos[:] = qpos_seq[t]
        mj.mj_forward(model, data)
        for j, bid in enumerate(body_ids):
            xpos[t, j] = data.xpos[bid]
    return xpos


def build_pairs_from_ik_config(ik_cfg, tables="both"):
    """
    从 bvh_to_g1.json 中构造 robot ↔ human 的“评估关节对”。
    返回：pairs: list[(robot_body_name, human_name, pos_weight)]
    """
    use_table1 = tables in ("1", "both") and ik_cfg.get("use_ik_match_table1", True)
    use_table2 = tables in ("2", "both") and ik_cfg.get("use_ik_match_table2", True)

    pairs = []
    seen = set()

    if use_table1 and "ik_match_table1" in ik_cfg:
        for robot_body_name, entry in ik_cfg["ik_match_table1"].items():
            human_name, pos_w, rot_w, pos_off, rot_off = entry
            if pos_w >= 0:
                key = (robot_body_name, human_name)
                if key not in seen:
                    pairs.append((robot_body_name, human_name, pos_w))
                    seen.add(key)

    if use_table2 and "ik_match_table2" in ik_cfg:
        for robot_body_name, entry in ik_cfg["ik_match_table2"].items():
            human_name, pos_w, rot_w, pos_off, rot_off = entry
            if pos_w > 0:
                key = (robot_body_name, human_name)
                if key not in seen:
                    pairs.append((robot_body_name, human_name, pos_w))
                    seen.add(key)

    if not pairs:
        raise ValueError("在 ik_match_table 中没有找到 pos_weight > 0 的关节，无法构造评估关节对。")
    return pairs


def _nan_stats(arr: np.ndarray) -> Dict[str, float]:
    """
    计算 min/median/mean/max（忽略 NaN）；若全 NaN 则全 NaN。
    """
    arr = np.asarray(arr, dtype=float).reshape(-1)
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return {"min": np.nan, "median": np.nan, "mean": np.nan, "max": np.nan}
    return {
        "min": float(np.min(valid)),
        "median": float(np.median(valid)),
        "mean": float(np.mean(valid)),
        "max": float(np.max(valid)),
    }


def _print_stats(prefix: str, stats: Dict[str, float]):
    print(f"{prefix}:")
    print(f"  min    = {stats['min']:.4f}")
    print(f"  median = {stats['median']:.4f}")
    print(f"  mean   = {stats['mean']:.4f}")
    print(f"  max    = {stats['max']:.4f}")


def evaluate_one_pkl(
    pkl_path: Path,
    model: mj.MjModel,
    pairs: List[Tuple[str, str, float]],
    robot_body_names: List[str],
    robot_body_ids: List[int],
    args,
) -> Dict[str, Any]:
    """
    评估单个 pkl，返回可序列化 dict。
    """
    # 1) 加载 pkl
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError("pkl 顶层不是 dict，无法解析。")
    if "robot_motion" not in data or "human_motion" not in data:
        raise ValueError("pkl 顶层缺少 robot_motion 或 human_motion。")

    robot_motion = data["robot_motion"]
    human_motion = data["human_motion"]

    # 2) fps
    fps_h = human_motion.get("fps", None)
    fps_r = robot_motion.get("fps", None)
    if fps_h is None and fps_r is None:
        raise ValueError("找不到 fps，请确认 pkl 保存时包含 'fps' 字段。")
    fps = float(fps_h if fps_h is not None else fps_r)

    # 3) 机器人 qpos
    qpos_robot = build_robot_qpos(robot_motion)  # (T_r, nq)

    # 4) 人体 anchor 世界坐标
    root_pos_h = np.asarray(human_motion["root_pos"], dtype=float)            # (T_h,3)
    local_body_pos = np.asarray(human_motion["local_body_pos"], dtype=float)  # (T_h,Nh,3)
    human_body_names = list(human_motion["link_body_list"])                   # 长度 Nh

    T_h, Nh, _ = local_body_pos.shape
    human_global_pos = local_body_pos + root_pos_h[:, None, :]                # (T_h,Nh,3)

    # 5) 由 pairs 确定 human_indices（每个文件可能都要映射一次）
    human_indices = []
    joint_pairs = []
    for robot_body_name, human_name, pos_w in pairs:
        if human_name not in human_body_names:
            raise ValueError(f"Human 关节 {human_name} 不在 human_motion['link_body_list'] 中。")
        human_indices.append(human_body_names.index(human_name))
        joint_pairs.append((robot_body_name, human_name))

    J = len(robot_body_names)

    # 6) 对齐时间长度
    T_r = qpos_robot.shape[0]
    T = int(min(T_r, T_h))
    qpos_robot = qpos_robot[:T]
    human_global_pos = human_global_pos[:T]

    # 机器人关节角序列（用于关节角平滑度）
    dof_pos = np.asarray(robot_motion["dof_pos"], dtype=float)[:T]  # (T,ndof)
    ndof = int(dof_pos.shape[1])

    # 7) FK 机器人 body 的世界坐标
    xpos_robot_all = fk_robot_bodies_by_id(model, qpos_robot, robot_body_ids)  # (T,J,3)

    # 8) 人体对应 anchor 的世界坐标
    xpos_human = np.zeros((T, J, 3), dtype=float)
    for j, idx_h in enumerate(human_indices):
        xpos_human[:, j, :] = human_global_pos[:, idx_h, :]

    # 9) 每个关节对分别计算位置 SPARC
    human_sparc_per_joint = np.zeros(J, dtype=float)
    robot_sparc_per_joint = np.zeros(J, dtype=float)

    for j in range(J):
        pos_h = xpos_human[:, j, :]
        pos_r = xpos_robot_all[:, j, :]
        human_sparc_per_joint[j] = sparc_from_3d_position(
            pos_h, fps, padlevel=4, fc_max=args.fc_max, amp_th=args.amp_th
        )
        robot_sparc_per_joint[j] = sparc_from_3d_position(
            pos_r, fps, padlevel=4, fc_max=args.fc_max, amp_th=args.amp_th
        )

    # 10) 机器人关节角的 SPARC / jerk RMS
    joint_sparc = np.zeros(ndof, dtype=float)
    joint_jerk = np.zeros(ndof, dtype=float)

    for j in range(ndof):
        qj = dof_pos[:, j]
        joint_sparc[j] = sparc_from_1d_signal(qj, fps, padlevel=4, fc_max=args.fc_max, amp_th=args.amp_th)
        joint_jerk[j] = jerk_rms_1d(qj, fps)

    # 汇总统计（便于 CSV/JSON）
    stats_h_pos = _nan_stats(human_sparc_per_joint)
    stats_r_pos = _nan_stats(robot_sparc_per_joint)
    stats_dof_sparc = _nan_stats(joint_sparc)
    stats_dof_jerk = _nan_stats(joint_jerk)

    result: Dict[str, Any] = {
        "file": str(pkl_path),
        "name": pkl_path.name,
        "fps": fps,
        "T": T,
        "J": J,
        "ndof": ndof,
        "human_pos_sparc": stats_h_pos,
        "robot_pos_sparc": stats_r_pos,
        "robot_angle_sparc": stats_dof_sparc,
        "robot_angle_jerk_rms": stats_dof_jerk,
    }

    if args.save_arrays:
        # 可能很大，默认关闭
        result["per_joint_pairs"] = joint_pairs
        result["human_pos_sparc_per_joint"] = human_sparc_per_joint.tolist()
        result["robot_pos_sparc_per_joint"] = robot_sparc_per_joint.tolist()
        result["robot_angle_sparc_per_dof"] = joint_sparc.tolist()
        result["robot_angle_jerk_rms_per_dof"] = joint_jerk.tolist()

    if args.detailed:
        print(f"\n================= {pkl_path} =================")
        print(f"[Info] 使用 fps = {fps} Hz")
        print("评估关节对（robot ↔ human）：")
        for j in range(J):
            rb, hn = joint_pairs[j]
            print(f"  {j:02d}: {rb}  <->  {hn}")

        if T < T_r or T < T_h:
            print(f"[Info] robot T={T_r}, human T={T_h}，对齐到 T={T}")

        print("\n=== 每个评估关节对的 位置 SPARC（越接近 0 越平滑） ===")
        for j in range(J):
            rb, hn = joint_pairs[j]
            print(f"[{j:02d}] {rb} <-> {hn}")
            print(f"     human SPARC = {human_sparc_per_joint[j]:.4f}")
            print(f"     robot SPARC = {robot_sparc_per_joint[j]:.4f}")

        print("\n=== 汇总统计（位置 SPARC） ===")
        _print_stats("Human joints (position SPARC)", stats_h_pos)
        _print_stats("Robot joints (position SPARC)", stats_r_pos)

        print("\n=== 机器人关节角平滑度：SPARC & jerk RMS ===")
        for j in range(ndof):
            print(f"[DOF {j:02d}] SPARC_angle = {joint_sparc[j]:.4f},  jerk_RMS = {joint_jerk[j]:.4e}")

        print("\n=== 汇总统计（机器人关节角 SPARC） ===")
        _print_stats("Robot DOFs (angle SPARC)", stats_dof_sparc)

        print("\n=== 汇总统计（机器人关节角 jerk RMS） ===")
        _print_stats("Robot DOFs (angle jerk RMS)", stats_dof_jerk)

        print("=================================================")
    else:
        # brief：每个文件一行
        print(
            f"[OK] {pkl_path.name} | T={T} fps={fps:g} | "
            f"posSPARC(h_mean={stats_h_pos['mean']:.4f}, r_mean={stats_r_pos['mean']:.4f}) | "
            f"angleSPARC(mean={stats_dof_sparc['mean']:.4f}) | "
            f"jerkRMS(mean={stats_dof_jerk['mean']:.4e})"
        )

    return result


def _collect_pkl_paths(pkl: Optional[str], pkl_dir: Optional[str], pattern: str, recursive: bool) -> List[Path]:
    paths: List[Path] = []
    if pkl:
        paths.append(Path(pkl))
    if pkl_dir:
        d = Path(pkl_dir)
        if recursive:
            paths.extend(sorted(d.rglob(pattern)))
        else:
            paths.extend(sorted(d.glob(pattern)))

    # 去重并保序
    seen = set()
    uniq: List[Path] = []
    for p in paths:
        sp = str(p)
        if sp not in seen:
            uniq.append(p)
            seen.add(sp)
    return uniq


def _flatten_for_csv(res: Dict[str, Any]) -> Dict[str, Any]:
    """
    把结果 dict 展平成 CSV 一行（只保留汇总统计）。
    """
    def add(prefix: str, stats: Dict[str, float], out: Dict[str, Any]):
        out[f"{prefix}_min"] = stats["min"]
        out[f"{prefix}_median"] = stats["median"]
        out[f"{prefix}_mean"] = stats["mean"]
        out[f"{prefix}_max"] = stats["max"]

    row: Dict[str, Any] = {
        "file": res.get("file", ""),
        "name": res.get("name", ""),
        "fps": res.get("fps", np.nan),
        "T": res.get("T", np.nan),
        "J": res.get("J", np.nan),
        "ndof": res.get("ndof", np.nan),
        "status": res.get("status", "ok"),
        "error": res.get("error", ""),
    }
    add("human_pos_sparc", res["human_pos_sparc"], row)
    add("robot_pos_sparc", res["robot_pos_sparc"], row)
    add("robot_angle_sparc", res["robot_angle_sparc"], row)
    add("robot_angle_jerk_rms", res["robot_angle_jerk_rms"], row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, default=None,
                        help="单个 pkl 文件（包含 robot_motion 和 human_motion）")
    parser.add_argument("--pkl_dir", type=str, default=None,
                        help="批量评估：pkl 文件夹路径（与 --pattern 配合）")
    parser.add_argument("--pattern", type=str, default="*.pkl",
                        help="批量评估：glob pattern（默认 '*.pkl'）")
    parser.add_argument("--recursive", action="store_true",
                        help="批量评估：递归搜索子目录（使用 rglob）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只评估前 N 个文件（用于快速检查）")

    parser.add_argument("--robot_xml", type=str, required=True,
                        help="MuJoCo 机器人 xml 路径（例如 g1_mocap_29dof.xml）")
    parser.add_argument("--ik_config", type=str, required=True,
                        help="IK 配置 json（例如 bvh_to_g1.json）")
    parser.add_argument("--tables", type=str, default="both",
                        choices=["1", "2", "both"],
                        help="使用 ik_match_table1 / 2 / both 中的哪些关节（默认 both）")

    parser.add_argument("--fc_max", type=float, default=10.0,
                        help="SPARC 计算时的最高考虑频率 fc_max (Hz)，默认 10.0")
    parser.add_argument("--amp_th", type=float, default=0.05,
                        help="SPARC 谱幅阈值 amp_th，默认 0.05")

    # 输出与控制
    parser.add_argument("--out_csv", type=str, default=None,
                        help="把每个文件的汇总结果保存为 CSV")
    parser.add_argument("--out_json", type=str, default=None,
                        help="把每个文件的完整结果保存为 JSON（可配合 --save_arrays）")
    parser.add_argument("--save_arrays", action="store_true",
                        help="JSON 中额外保存 per-joint / per-dof 的数组（文件会很大）")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="遇到某个文件报错时继续下一个文件（默认关闭时会直接 raise）")

    # detailed：默认策略 —— 单文件默认 True，批量默认 False
    parser.add_argument("--detailed", action=argparse.BooleanOptionalAction, default=None,
                        help="是否打印逐关节/逐DOF的详细信息（单文件默认开启；批量默认关闭）")

    args = parser.parse_args()

    # 自动设置 detailed 默认值（保持单文件行为接近原版，批量则更清爽）
    if args.detailed is None:
        args.detailed = (args.pkl is not None and args.pkl_dir is None)

    # 收集输入文件
    pkl_paths = _collect_pkl_paths(args.pkl, args.pkl_dir, args.pattern, args.recursive)
    if len(pkl_paths) == 0:
        raise ValueError("未找到任何 pkl：请提供 --pkl 或 --pkl_dir（并确认 --pattern 匹配）")

    for p in pkl_paths:
        if not p.exists():
            raise FileNotFoundError(p)

    if args.limit is not None:
        pkl_paths = pkl_paths[: max(0, int(args.limit))]

    batch_mode = (args.pkl_dir is not None) or (len(pkl_paths) > 1)
    if batch_mode:
        print(f"[Info] Batch mode: {len(pkl_paths)} files | pattern={args.pattern} | recursive={args.recursive} | detailed={args.detailed}")
    else:
        print(f"[Info] Single mode: {pkl_paths[0]} | detailed={args.detailed}")

    # 预加载 MuJoCo 模型 & IK 配置（批量时避免重复开销）
    model = mj.MjModel.from_xml_path(args.robot_xml)
    with open(args.ik_config, "r", encoding="utf-8") as f:
        ik_cfg = json.load(f)
    pairs = build_pairs_from_ik_config(ik_cfg, tables=args.tables)

    # 预先从 pairs 得到 robot_body_names，并在模型里做一次 name2id
    robot_body_names = [rb for (rb, hn, pw) in pairs]
    robot_body_ids: List[int] = []
    for rb in robot_body_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, rb)
        if bid < 0:
            raise ValueError(f"Body '{rb}' not found in model: {args.robot_xml}")
        robot_body_ids.append(int(bid))

    # 逐文件评估
    results: List[Dict[str, Any]] = []
    n_ok, n_fail = 0, 0

    for i, pkl_path in enumerate(pkl_paths):
        try:
            res = evaluate_one_pkl(
                pkl_path=pkl_path,
                model=model,
                pairs=pairs,
                robot_body_names=robot_body_names,
                robot_body_ids=robot_body_ids,
                args=args,
            )
            res["status"] = "ok"
            results.append(res)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            err = f"{type(e).__name__}: {e}"
            if args.continue_on_error:
                print(f"[FAIL] {pkl_path.name} | {err}")
                if args.detailed:
                    print(traceback.format_exc())
                results.append({
                    "file": str(pkl_path),
                    "name": pkl_path.name,
                    "status": "fail",
                    "error": err,
                    "human_pos_sparc": {"min": np.nan, "median": np.nan, "mean": np.nan, "max": np.nan},
                    "robot_pos_sparc": {"min": np.nan, "median": np.nan, "mean": np.nan, "max": np.nan},
                    "robot_angle_sparc": {"min": np.nan, "median": np.nan, "mean": np.nan, "max": np.nan},
                    "robot_angle_jerk_rms": {"min": np.nan, "median": np.nan, "mean": np.nan, "max": np.nan},
                })
                continue
            raise

    print(f"\n[Done] ok={n_ok}, fail={n_fail}, total={len(results)}")

    # Overall 汇总（基于每个文件的 mean 再做一次统计）
    if n_ok > 0:
        def collect_means(key: str) -> np.ndarray:
            vals = []
            for r in results:
                if r.get("status") == "ok":
                    vals.append(r[key]["mean"])
            return np.asarray(vals, dtype=float)

        overall = {
            "human_pos_sparc_mean_over_files": _nan_stats(collect_means("human_pos_sparc")),
            "robot_pos_sparc_mean_over_files": _nan_stats(collect_means("robot_pos_sparc")),
            "robot_angle_sparc_mean_over_files": _nan_stats(collect_means("robot_angle_sparc")),
            "robot_angle_jerk_rms_mean_over_files": _nan_stats(collect_means("robot_angle_jerk_rms")),
        }

        print("\n=== Overall（跨文件：取每个文件的 mean 再统计） ===")
        _print_stats("Human pos SPARC mean over files", overall["human_pos_sparc_mean_over_files"])
        _print_stats("Robot pos SPARC mean over files", overall["robot_pos_sparc_mean_over_files"])
        _print_stats("Robot angle SPARC mean over files", overall["robot_angle_sparc_mean_over_files"])
        _print_stats("Robot jerk RMS mean over files", overall["robot_angle_jerk_rms_mean_over_files"])
    else:
        overall = {}

    # 导出 CSV
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [_flatten_for_csv(r) for r in results]
        fieldnames = list(rows[0].keys()) if rows else []
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Saved] CSV -> {out_csv}")

    # 导出 JSON
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "args": vars(args),
            "summary": {"ok": n_ok, "fail": n_fail, "total": len(results)},
            "overall": overall,
            "results": results,
        }
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[Saved] JSON -> {out_json}")


if __name__ == "__main__":
    main()


"""
python tools/eval_smoothness_extended_batch.py \
  --pkl_dir retargeting_result/multi_robot_net_results/n1 \
  --pattern "*.pkl" \
  --robot_xml assets/fourier_n1/n1_mocap.xml \
  --ik_config general_motion_retargeting/ik_configs/bvh_to_n1.json \
  --tables both \
  --out_csv retargeting_result/baseline/PHC/g1/smoothness_summary.csv
"""