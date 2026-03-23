import argparse
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_lafan1_file


# ---------------- Quaternion helpers (wxyz) ----------------
def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=-1)


def quat_inv_wxyz(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q)
    conj = quat_conj_wxyz(q)
    norm2 = np.sum(q * q, axis=-1, keepdims=True)
    return conj / np.maximum(norm2, eps)


def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz)
    return q_wxyz[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(q_xyzw)
    return q_xyzw[..., [3, 0, 1, 2]]


def build_human_target_list(retargeter: GMR) -> List[str]:
    """
    从 ik_match_table1/2 中提取 human target joints 的稳定顺序（按 table 遍历顺序，去重）。
    只取 pos_weight!=0 或 rot_weight!=0 的项（真正作为 target 的）。
    """
    names: List[str] = []

    def add_from_table(tbl):
        for _, entry in tbl.items():
            body_name, pos_w, rot_w, _, _ = entry
            if (pos_w != 0) or (rot_w != 0):
                if body_name not in names:
                    names.append(body_name)

    if getattr(retargeter, "use_ik_match_table1", True):
        add_from_table(retargeter.ik_match_table1)
    if getattr(retargeter, "use_ik_match_table2", False):
        add_from_table(retargeter.ik_match_table2)

    # 确保 root 在列表里且可定位
    root_name = retargeter.human_root_name
    if root_name not in names:
        names = [root_name] + names

    return names


def extract_targets_from_scaled_human_data(scaled_human_data: dict, body_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 retargeter.scaled_human_data（wxyz）按 body_names 顺序抽取 (pos, quat_xyzw)
    返回:
      pos: (N,3)
      rot_xyzw: (N,4)
    """
    pos = []
    rot_xyzw = []
    for n in body_names:
        if n not in scaled_human_data:
            raise KeyError(
                f"scaled_human_data missing key '{n}'. Available keys: {list(scaled_human_data.keys())[:10]} ..."
            )
        p, q_wxyz = scaled_human_data[n]
        pos.append(np.asarray(p, dtype=np.float32))
        rot_xyzw.append(wxyz_to_xyzw(np.asarray(q_wxyz, dtype=np.float32)))
    return np.stack(pos, axis=0), np.stack(rot_xyzw, axis=0)


def compute_root_relative(pos_global: np.ndarray, rot_global_xyzw: np.ndarray, root_idx: int):
    """
    pos_global: (T,N,3)  world
    rot_global_xyzw: (T,N,4) world
    返回:
      root_pos (T,3), root_rot_xyzw (T,4),
      local_pos (T,N,3), local_rot_xyzw (T,N,4)  (root-relative)
    """
    root_pos = pos_global[:, root_idx]
    root_rot = rot_global_xyzw[:, root_idx]

    local_pos = pos_global - root_pos[:, None, :]

    root_wxyz = xyzw_to_wxyz(root_rot)                # (T,4)
    joint_wxyz = xyzw_to_wxyz(rot_global_xyzw)        # (T,N,4)
    root_inv = quat_inv_wxyz(root_wxyz)[:, None, :]   # (T,1,4)
    local_wxyz = quat_mul_wxyz(root_inv, joint_wxyz)
    local_rot_xyzw = wxyz_to_xyzw(local_wxyz).astype(np.float32)

    return (
        root_pos.astype(np.float32),
        root_rot.astype(np.float32),
        local_pos.astype(np.float32),
        local_rot_xyzw,
    )


def process_one_bvh(
    bvh_file: str,
    robot: str,
    save_path: str,
    save_human_gmr: bool = False,
    offset_to_ground: bool = True,
) -> None:
    """
    处理单个 BVH -> 输出 pkl（格式与原脚本一致）。
    """
    save_path = str(save_path)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # Load BVH frames
    lafan1_data_frames, actual_human_height = load_lafan1_file(bvh_file)
    motion_fps = 30

    # retargeter_label: 正常GMR（生成 robot_motion teacher）
    retargeter_label = GMR(src_human="bvh", tgt_robot=robot, actual_human_height=actual_human_height)

    # retargeter_scale1: 同一套json（含 rot_offset/pos_offset/ground），但 scale 置1
    retargeter_scale1 = GMR(src_human="bvh", tgt_robot=robot, actual_human_height=actual_human_height)
    retargeter_scale1.human_scale_table = {k: 1.0 for k in retargeter_scale1.human_scale_table.keys()}

    # human target joints 顺序（固定）
    body_names = build_human_target_list(retargeter_label)
    root_name = retargeter_label.human_root_name
    root_idx = body_names.index(root_name)

    # Buffers
    qpos_list = []

    # scale=1 aligned (仍保留 rot_offset/pos_offset/ground)
    h1_pos_list = []
    h1_rot_list = []

    # optional: gmr scaled
    hg_pos_list = []
    hg_rot_list = []

    for frame in tqdm(lafan1_data_frames, desc=f"Retarget {Path(bvh_file).name}", leave=False):
        # 1) teacher qpos（会更新 retargeter_label.scaled_human_data）
        qpos = retargeter_label.retarget(frame, offset_to_ground=offset_to_ground)
        qpos_list.append(qpos)

        # 2) scale=1 aligned human targets（仅update_targets，不解IK）
        retargeter_scale1.update_targets(frame, offset_to_ground=offset_to_ground)
        pos1, rot1 = extract_targets_from_scaled_human_data(retargeter_scale1.scaled_human_data, body_names)
        h1_pos_list.append(pos1)
        h1_rot_list.append(rot1)

        # 3) 可选保存正常GMR的 human targets（用于debug/消融）
        if save_human_gmr:
            posg, rotg = extract_targets_from_scaled_human_data(retargeter_label.scaled_human_data, body_names)
            hg_pos_list.append(posg)
            hg_rot_list.append(rotg)

    # ---------------- robot_motion ----------------
    qpos_all = np.stack(qpos_list, axis=0).astype(np.float32)  # (T, 7+DoF)
    robot_motion = {
        "fps": motion_fps,
        "root_pos": qpos_all[:, :3],
        "root_rot": qpos_all[:, 3:7][:, [1, 2, 3, 0]],  # wxyz->xyzw
        "dof_pos": qpos_all[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }

    # ---------------- human_motion_scale1 ----------------
    h1_pos = np.stack(h1_pos_list, axis=0).astype(np.float32)  # (T,N,3)
    h1_rot = np.stack(h1_rot_list, axis=0).astype(np.float32)  # (T,N,4) xyzw

    h1_root_pos, h1_root_rot, h1_local_pos, h1_local_rot = compute_root_relative(h1_pos, h1_rot, root_idx)

    human_motion_scale1 = {
        "fps": motion_fps,
        "root_pos": h1_root_pos,
        "root_rot": h1_root_rot,
        "global_body_pos": h1_pos,         # (T,N,3) world，但已应用 rot_offset/pos_offset/ground
        "global_body_rot": h1_rot,         # (T,N,4) world，但已应用 rot_offset（坐标系对齐）
        "local_body_pos": h1_local_pos,    # (T,N,3) root-translation removed
        "local_body_rot": h1_local_rot,    # (T,N,4) root-relative rotations
        "link_body_list": body_names,
        "note": "scale=1 (human_scale_table all ones), but still applies json rot_offset/pos_offset/ground via update_targets()",
    }

    save_data = {
        "robot_motion": robot_motion,
        "human_motion_scale1": human_motion_scale1,
    }

    # ---------------- optional human_motion_gmr ----------------
    if save_human_gmr:
        hg_pos = np.stack(hg_pos_list, axis=0).astype(np.float32)
        hg_rot = np.stack(hg_rot_list, axis=0).astype(np.float32)
        hg_root_pos, hg_root_rot, hg_local_pos, hg_local_rot = compute_root_relative(hg_pos, hg_rot, root_idx)
        save_data["human_motion"] = {
            "fps": motion_fps,
            "root_pos": hg_root_pos,
            "root_rot": hg_root_rot,
            "global_body_pos": hg_pos,
            "global_body_rot": hg_rot,
            "local_body_pos": hg_local_pos,
            "local_body_rot": hg_local_rot,
            "link_body_list": body_names,
            "note": "normal GMR scaled_human_data (uses human_scale_table)",
        }

    with open(save_path, "wb") as f:
        pickle.dump(save_data, f)


def list_bvh_files(bvh_dir: str, pattern: str = "*.bvh", recursive: bool = False) -> List[Path]:
    root = Path(bvh_dir)
    if not root.exists():
        raise FileNotFoundError(f"bvh_dir not found: {bvh_dir}")
    if recursive:
        files = sorted(root.rglob(pattern))
    else:
        files = sorted(root.glob(pattern))
    files = [p for p in files if p.is_file()]
    return files


def make_out_path(
    bvh_path: Path,
    bvh_dir: Path,
    save_dir: Path,
    keep_tree: bool,
) -> Path:
    if keep_tree:
        rel = bvh_path.relative_to(bvh_dir).with_suffix(".pkl")
        out = save_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        return out
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / f"{bvh_path.stem}.pkl"


def main():
    parser = argparse.ArgumentParser()

    # -------- input (single vs batch) --------
    io_group = parser.add_mutually_exclusive_group(required=True)
    io_group.add_argument("--bvh_file", type=str, help="单个 BVH 文件路径")
    io_group.add_argument("--bvh_dir", type=str, help="BVH 文件夹（批量处理）")

    # single output
    parser.add_argument("--save_path", type=str, default=None, help="单文件模式输出 pkl 路径（与 --bvh_file 搭配）")

    # batch output
    parser.add_argument("--save_dir", type=str, default=None, help="批量模式输出目录（与 --bvh_dir 搭配）")
    parser.add_argument("--pattern", type=str, default="*.bvh", help="批量模式匹配模式（默认 *.bvh）")
    parser.add_argument("--recursive", action="store_true", help="批量模式递归搜索子目录")
    parser.add_argument("--keep_tree", action="store_true", help="批量模式保持目录结构（需要 --recursive 或希望复用层级）")
    parser.add_argument("--skip_existing", action="store_true", help="若输出 pkl 已存在则跳过")
    parser.add_argument("--fail_fast", action="store_true", help="遇到错误立刻退出（默认继续处理其他文件）")

    # -------- core options --------
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "booster_t1_29dof",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "zlite",
        ],
        default="unitree_g1",
    )
    parser.add_argument(
        "--save_human_gmr",
        action="store_true",
        help="额外保存一份 human_motion（正常GMR scaled_human_data）用于对比/消融",
    )

    # Default True but allow disable
    parser.set_defaults(offset_to_ground=True)
    parser.add_argument(
        "--no_offset_to_ground",
        dest="offset_to_ground",
        action="store_false",
        help="关闭 offset_human_data_to_ground（默认开启）",
    )

    args = parser.parse_args()

    # -------- single file mode --------
    if args.bvh_file is not None:
        if not args.save_path:
            raise ValueError("Single-file mode requires --save_path together with --bvh_file.")
        process_one_bvh(
            bvh_file=args.bvh_file,
            robot=args.robot,
            save_path=args.save_path,
            save_human_gmr=args.save_human_gmr,
            offset_to_ground=args.offset_to_ground,
        )
        print(f"[OK] Saved to: {args.save_path}")
        return

    # -------- batch mode --------
    if args.bvh_dir is not None:
        if not args.save_dir:
            raise ValueError("Batch mode requires --save_dir together with --bvh_dir.")

        bvh_dir = Path(args.bvh_dir)
        save_dir = Path(args.save_dir)

        files = list_bvh_files(args.bvh_dir, pattern=args.pattern, recursive=args.recursive)
        if len(files) == 0:
            raise FileNotFoundError(f"No files matched: dir={args.bvh_dir}, pattern={args.pattern}, recursive={args.recursive}")

        errors = []
        for bvh_path in tqdm(files, desc="Batch BVH -> PKL"):
            out_path = make_out_path(bvh_path, bvh_dir, save_dir, keep_tree=args.keep_tree)

            if args.skip_existing and out_path.exists():
                continue

            try:
                process_one_bvh(
                    bvh_file=str(bvh_path),
                    robot=args.robot,
                    save_path=str(out_path),
                    save_human_gmr=args.save_human_gmr,
                    offset_to_ground=args.offset_to_ground,
                )
            except Exception as e:
                msg = f"[ERR] {bvh_path} -> {out_path} : {type(e).__name__}: {e}"
                errors.append(msg)
                if args.fail_fast:
                    raise
                else:
                    print(msg)

        print(f"\n[OK] Done. total={len(files)}, failed={len(errors)}, saved_dir={save_dir}")
        if errors:
            print("\n--- Errors (first 20) ---")
            for m in errors[:20]:
                print(m)
        return


if __name__ == "__main__":
    main()


"""
python scripts/bvh_to_robot_save_scale1_human_batch.py \
  --bvh_dir lafan1  \
  --save_dir retargeting_result/pm01_input \
  --pattern "*.bvh" \
  --robot engineai_pm01 \
  --save_human_gmr \
  --skip_existing
"""