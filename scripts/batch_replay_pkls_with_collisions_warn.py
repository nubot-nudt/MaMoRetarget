#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
批量回放文件夹内的 pkl（bvh_to_robot_save.py / refine_retargeting.py 输出），
用 RobotMotionViewer_warn 检测自碰撞，并把每个文件的统计信息写入一个 txt。

基于 replay_pkl_with_collisions_warn.py 的逻辑：
- root_rot: xyzw -> wxyz
- viewer.step(..., return_self_collision_contacts=True) 拿到每帧 contacts
- 汇总 frames_with_collision / total_contacts / dist/penetration 等
"""

import argparse
import pathlib
import pickle
import time
import traceback
from typing import Dict, Any, List, Tuple

import numpy as np
from tqdm import tqdm

from general_motion_retargeting import RobotMotionViewer_warn


# -----------------------------
# from your original script
# -----------------------------
def load_motion_pkl(pkl_path: pathlib.Path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError("pkl 顶层不是 dict，无法解析。")
    if "robot_motion" not in data:
        raise ValueError("pkl 中没有 'robot_motion' 字段。")

    robot_motion = data["robot_motion"]
    human_motion = data.get("human_motion", None)
    return robot_motion, human_motion


def reconstruct_human_dict_for_frame(human_motion, frame_idx):
    if human_motion is None:
        return None

    root_pos = np.asarray(human_motion["root_pos"], dtype=float)
    root_rot_xyzw = np.asarray(human_motion["root_rot"], dtype=float)
    local_body_pos = np.asarray(human_motion["local_body_pos"], dtype=float)
    body_names = list(human_motion["link_body_list"])

    T, N, _ = local_body_pos.shape
    if frame_idx < 0 or frame_idx >= T:
        return None

    root_p = root_pos[frame_idx]
    root_q_xyzw = root_rot_xyzw[frame_idx]
    root_q_wxyz = np.array(
        [root_q_xyzw[3], root_q_xyzw[0], root_q_xyzw[1], root_q_xyzw[2]],
        dtype=float,
    )

    global_pos = local_body_pos[frame_idx] + root_p[None, :]
    scaled_human = {}
    for j, name in enumerate(body_names):
        scaled_human[name] = (global_pos[j].copy(), root_q_wxyz.copy())
    return scaled_human


# -----------------------------
# batch helpers
# -----------------------------
def list_pkl_files(pkl_dir: pathlib.Path, pattern: str, recursive: bool) -> List[pathlib.Path]:
    if recursive:
        files = sorted(pkl_dir.rglob(pattern))
    else:
        files = sorted(pkl_dir.glob(pattern))
    return [p for p in files if p.is_file()]


def compute_collision_stats(collision_log: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    total_frames = len(collision_log)
    frames_with_collision = sum(1 for fc in collision_log if fc)
    total_contacts = sum(len(fc) for fc in collision_log)
    mean_contacts_per_frame = total_contacts / max(1, total_frames)

    # flatten with frame index for debugging
    flat: List[Tuple[int, Dict[str, Any]]] = []
    dists: List[float] = []
    for fi, fc in enumerate(collision_log):
        for c in fc:
            dist = c.get("dist", None)
            if dist is None:
                continue
            d = float(dist)
            dists.append(d)
            flat.append((fi, c))

    if dists:
        arr = np.asarray(dists, dtype=float)
        mean_dist = float(np.mean(arr))
        min_dist = float(np.min(arr))
        max_dist = float(np.max(arr))

        penetrations = (-arr[arr < 0.0])  # positive depth
        mean_penetration = float(np.mean(penetrations)) if penetrations.size > 0 else 0.0
        frac_penetrating = float(penetrations.size / arr.size)

        per_frame_mean = [float(np.mean([cc["dist"] for cc in fc if "dist" in cc])) for fc in collision_log if fc]
        per_collision_frame_mean_dist = float(np.mean(per_frame_mean)) if per_frame_mean else None

        # top-K deepest penetrations (most negative dist)
        flat_sorted = sorted(
            [(fi, c) for fi, c in flat if float(c.get("dist", 0.0)) < 0.0],
            key=lambda x: float(x[1]["dist"])
        )
        topk = flat_sorted[:5]
    else:
        mean_dist = min_dist = max_dist = 0.0
        mean_penetration = 0.0
        frac_penetrating = 0.0
        per_collision_frame_mean_dist = None
        topk = []

    return dict(
        total_frames=total_frames,
        frames_with_collision=frames_with_collision,
        total_contacts=total_contacts,
        mean_contacts_per_frame=mean_contacts_per_frame,
        mean_dist=mean_dist,
        min_dist=min_dist,
        max_dist=max_dist,
        mean_penetration=mean_penetration,
        frac_penetrating=frac_penetrating,
        per_collision_frame_mean_dist=per_collision_frame_mean_dist,
        top5_deepest_contacts=topk,  # list[(frame_idx, contact_dict)]
    )


def process_one_file(pkl_path: pathlib.Path, args) -> Dict[str, Any]:
    robot_motion, human_motion = load_motion_pkl(pkl_path)

    fps_r = robot_motion.get("fps", None)
    fps_h = human_motion.get("fps", None) if human_motion is not None else None
    if fps_r is None and fps_h is None:
        raise ValueError("pkl 中 robot_motion 和 human_motion 都没有 'fps' 字段。")
    motion_fps = fps_r if fps_r is not None else fps_h

    root_pos = np.asarray(robot_motion["root_pos"], dtype=float)       # (T,3)
    root_rot_xyzw = np.asarray(robot_motion["root_rot"], dtype=float)  # (T,4) xyzw
    dof_pos = np.asarray(robot_motion["dof_pos"], dtype=float)         # (T,ndof)

    T = root_pos.shape[0]
    if root_rot_xyzw.shape[0] != T or dof_pos.shape[0] != T:
        raise ValueError("robot_motion 中 root_pos / root_rot / dof_pos 的帧数不一致。")
    if args.max_frames > 0:
        T = min(T, args.max_frames)

    # per-file video path (avoid overwrite)
    video_path = None
    if args.record_video:
        args.video_dir.mkdir(parents=True, exist_ok=True)
        video_path = args.video_dir / (pkl_path.stem + ".mp4")

    viewer = None
    collision_log: List[List[Dict[str, Any]]] = []
    try:
        viewer = RobotMotionViewer_warn(
            robot_type=args.robot,
            motion_fps=motion_fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=str(video_path) if (args.record_video and video_path is not None) else None,
        )

        for i in range(T):
            rr_xyzw = root_rot_xyzw[i]
            rr_wxyz = np.array([rr_xyzw[3], rr_xyzw[0], rr_xyzw[1], rr_xyzw[2]], dtype=float)

            if args.show_human and human_motion is not None:
                human_dict = reconstruct_human_dict_for_frame(human_motion, i)
            else:
                human_dict = None

            frame_contacts = viewer.step(
                root_pos=root_pos[i],
                root_rot=rr_wxyz,
                dof_pos=dof_pos[i],
                human_motion_data=human_dict,
                rate_limit=args.rate_limit,
                highlight_self_collision=args.highlight_self_collision,
                return_self_collision_contacts=True,
            )
            collision_log.append(frame_contacts)

    finally:
        if viewer is not None:
            viewer.close()

    stats = compute_collision_stats(collision_log)
    stats.update(
        dict(
            file=str(pkl_path),
            fps=float(motion_fps),
            used_frames=int(T),
            recorded_video=str(video_path) if video_path is not None else None,
        )
    )
    return stats


def format_stats_block(stats: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"FILE: {stats['file']}")
    lines.append(f"  fps = {stats['fps']}")
    lines.append(f"  used_frames = {stats['used_frames']}")
    if stats.get("recorded_video"):
        lines.append(f"  video = {stats['recorded_video']}")

    lines.append(f"  [Collision Summary] {stats['frames_with_collision']}/{stats['total_frames']} frames have self-collision.")
    lines.append(f"  [Collision Detail] total_contacts = {stats['total_contacts']}, mean_contacts/frame = {stats['mean_contacts_per_frame']:.3f}")
    lines.append(
        f"  [Collision Dist] mean(dist) = {stats['mean_dist']:.6f} m | "
        f"min(dist) = {stats['min_dist']:.6f} m | max(dist) = {stats['max_dist']:.6f} m"
    )
    lines.append(
        f"  [Collision Penetration] mean(penetration) = {stats['mean_penetration']:.6f} m | "
        f"penetrating fraction = {stats['frac_penetrating']:.3%}"
    )
    if stats.get("per_collision_frame_mean_dist") is not None:
        lines.append(
            f"  [Collision Dist Per-Collision-Frame] mean = {stats['per_collision_frame_mean_dist']:.6f} m "
            f"(each collision frame equally weighted)"
        )

    topk = stats.get("top5_deepest_contacts", [])
    if topk:
        lines.append("  [Top-5 Deepest Penetrations] (most negative dist)")
        for rank, (fi, c) in enumerate(topk, 1):
            lines.append(
                f"    #{rank} frame={fi} dist={float(c['dist']):.6f} | "
                f"{c.get('body1','?')} <-> {c.get('body2','?')} | "
                f"geom: {c.get('geom1','?')} vs {c.get('geom2','?')}"
            )
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl_dir", type=str, required=True, help="包含 pkl 的文件夹")
    p.add_argument("--pattern", type=str, default="*.pkl", help="匹配模式（默认 *.pkl）")
    p.add_argument("--recursive", action="store_true", default=False, help="是否递归搜索子目录")

    p.add_argument("--out_txt", type=str, default="collision_batch_stats.txt", help="统计输出 txt（默认写到 pkl_dir 下）")

    p.add_argument("--robot", default="unitree_g1",
                   choices=[
                       "unitree_g1",
                       "unitree_g1_with_hands",
                       "booster_t1_29dof",
                       "stanford_toddy",
                       "fourier_n1",
                       "engineai_pm01",
                       "zlite",
                   ])
    p.add_argument("--rate_limit", action="store_true", default=False)
    p.add_argument("--show_human", action="store_true", default=False)

    p.add_argument("--highlight_self_collision", action="store_true", default=True)
    p.add_argument("--no_highlight_self_collision", action="store_false", dest="highlight_self_collision")

    # 批量默认不录视频，避免慢 & 覆盖
    p.add_argument("--record_video", action="store_true", default=False)
    p.add_argument("--video_dir", type=str, default="videos_batch", help="若录视频，输出目录（默认 videos_batch）")

    p.add_argument("--max_frames", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    pkl_dir = pathlib.Path(args.pkl_dir)
    if not pkl_dir.exists():
        raise FileNotFoundError(f"pkl_dir not found: {pkl_dir}")

    args.video_dir = pathlib.Path(args.video_dir)

    out_txt = pathlib.Path(args.out_txt)
    if not out_txt.is_absolute():
        out_txt = pkl_dir / out_txt

    files = list_pkl_files(pkl_dir, args.pattern, args.recursive)
    if not files:
        raise RuntimeError(f"在 {pkl_dir} 下未找到匹配 {args.pattern} 的文件（recursive={args.recursive}）")

    header = []
    header.append("=== Batch Self-Collision Stats ===")
    header.append(f"pkl_dir: {str(pkl_dir)}")
    header.append(f"pattern: {args.pattern} | recursive: {args.recursive}")
    header.append(f"robot: {args.robot}")
    header.append(f"rate_limit: {args.rate_limit} | show_human: {args.show_human} | highlight: {args.highlight_self_collision}")
    header.append(f"record_video: {args.record_video} | video_dir: {args.video_dir if args.record_video else 'N/A'}")
    header.append(f"max_frames: {args.max_frames}")
    header.append(f"num_files: {len(files)}")
    header.append("")

    ok, fail = 0, 0
    t0 = time.time()

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")

        for pkl_path in tqdm(files, desc="Batch processing", ncols=100):
            f.write("\n" + ("-" * 80) + "\n")
            start = time.time()
            try:
                stats = process_one_file(pkl_path, args)
                dt = time.time() - start
                f.write(format_stats_block(stats) + "\n")
                f.write(f"  runtime_sec = {dt:.3f}\n")
                ok += 1
            except Exception as e:
                dt = time.time() - start
                f.write(f"FILE: {str(pkl_path)}\n")
                f.write(f"  [ERROR] {repr(e)}\n")
                f.write(f"  runtime_sec = {dt:.3f}\n")
                f.write("  traceback:\n")
                f.write(traceback.format_exc() + "\n")
                fail += 1

        total_dt = time.time() - t0
        f.write("\n" + ("=" * 80) + "\n")
        f.write(f"Done. ok={ok}, fail={fail}, total_sec={total_dt:.3f}\n")

    print(f"[DONE] wrote stats to: {out_txt} | ok={ok}, fail={fail}")


if __name__ == "__main__":
    main()


"""
python scripts/batch_replay_pkls_with_collisions_warn.py \
  --pkl_dir retargeting_result/baseline/PM/t1 \
  --robot booster_t1_29dof \
  --show_human \
  --rate_limit \
  --out_txt z_collision_batch_stats.txt
"""