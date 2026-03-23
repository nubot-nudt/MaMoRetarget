#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回放 bvh_to_robot_save.py / refine_retargeting.py 生成的 pkl，
用改好版 RobotMotionViewer 可视化，并在“碰撞帧”显示明显标识（变红 + contact点/力），
同时统计自碰撞 dist/penetration。

注意：
- pkl 中保存的 root_rot 是 xyzw
- RobotMotionViewer.step 期望 root_rot 是 wxyz（mujoco 标量在前）
- 本脚本回放时会做 xyzw -> wxyz 转换
"""

import argparse
import pathlib
import pickle
import numpy as np
from rich import print
from tqdm import tqdm

from general_motion_retargeting import RobotMotionViewer_warn


def load_motion_pkl(pkl_path: pathlib.Path):
    """加载 bvh_to_robot_save.py / refine_retargeting.py 保存的 pkl。"""
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
    """
    用 human_motion 中的 root_pos + local_body_pos 重建一帧的 human_motion_data，
    供 RobotMotionViewer.step 使用。

    这里 human_motion 的结构来自 bvh_to_robot_save.py：
        human_motion = {
            "fps": motion_fps,
            "root_pos": (T,3),
            "root_rot": (T,4)  # xyzw
            "dof_pos": None,
            "local_body_pos": (T,N,3),
            "link_body_list": [N个关节名],
        }

    但没有保存每个关节自己的旋转四元数，因此这里只能近似地：
        - 所有关节旋转都用 root_rot（wxyz）
    这对你调试机器人碰撞是没问题的，因为碰撞只看 robot。
    """
    if human_motion is None:
        return None

    root_pos = np.asarray(human_motion["root_pos"], dtype=float)
    root_rot_xyzw = np.asarray(human_motion["root_rot"], dtype=float)
    local_body_pos = np.asarray(human_motion["local_body_pos"], dtype=float)
    body_names = list(human_motion["link_body_list"])

    T, N, _ = local_body_pos.shape
    if frame_idx < 0 or frame_idx >= T:
        return None

    root_p = root_pos[frame_idx]           # (3,)
    root_q_xyzw = root_rot_xyzw[frame_idx] # (4,)
    # 保存时是 xyzw，这里转回 wxyz 以匹配 RobotMotionViewer 的用法
    root_q_wxyz = np.array(
        [root_q_xyzw[3], root_q_xyzw[0], root_q_xyzw[1], root_q_xyzw[2]],
        dtype=float,
    )

    # local_body_pos 是相对 root 的位移
    global_pos = local_body_pos[frame_idx] + root_p[None, :]   # (N,3)

    scaled_human = {}
    for j, name in enumerate(body_names):
        # 旋转近似用 root 的旋转（wxyz）
        scaled_human[name] = (global_pos[j].copy(), root_q_wxyz.copy())

    return scaled_human


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl_file", type=str, required=True,
                   help="bvh_to_robot_save.py / refine_retargeting.py 保存的 pkl 路径")
    p.add_argument("--robot", default="unitree_g1",
                   choices=[
                       "unitree_g1",
                       "unitree_g1_with_hands",
                       "booster_t1",
                       "stanford_toddy",
                       "fourier_n1",
                       "engineai_pm01",
                       "zlite",
                   ])
    p.add_argument("--rate_limit", action="store_true", default=False,
                   help="是否按 fps 控制播放速度（True=实时，False=尽快播放）")
    p.add_argument("--show_human", action="store_true", default=False,
                   help="是否在 viewer 中同时渲染 human（用 human_motion 重建，旋转近似）")

    # --- 可视化需求：碰撞帧红色标识 ---
    p.add_argument("--highlight_self_collision", action="store_true", default=True,
                   help="碰撞帧高亮（机器人变红 + contact 点/力）")
    p.add_argument("--no_highlight_self_collision", action="store_false",
                   dest="highlight_self_collision",
                   help="关闭碰撞帧高亮（调试用）")

    # --- 录制视频（由 RobotMotionViewer 负责） ---
    p.add_argument("--record_video", action="store_true", default=True)
    p.add_argument("--video_path", type=str,
                   default="/home/nubot/GMR_old/videos/aiming1_subject5_warn.mp4")

    # --- 调试用：限制帧数 ---
    p.add_argument("--max_frames", type=int, default=-1,
                   help="限制回放帧数（-1 表示全部）")

    # --- 打印每帧碰撞细节 ---
    p.add_argument("--print_collision_frames", action="store_true", default=False,
                   help="打印每个碰撞帧的 contact 详情（帧多时会很刷屏）")
    return p.parse_args()


def main():
    args = parse_args()
    pkl_path = pathlib.Path(args.pkl_file)
    if not pkl_path.exists():
        raise FileNotFoundError(f"pkl not found: {pkl_path}")

    print(f"[cyan]加载 pkl:[/cyan] {pkl_path}")
    robot_motion, human_motion = load_motion_pkl(pkl_path)

    # fps：优先 robot_motion，其次 human_motion
    fps_r = robot_motion.get("fps", None)
    fps_h = human_motion.get("fps", None) if human_motion is not None else None
    if fps_r is None and fps_h is None:
        raise ValueError("pkl 中 robot_motion 和 human_motion 都没有 'fps' 字段。")

    motion_fps = fps_r if fps_r is not None else fps_h
    print(f"[cyan]使用 fps = {motion_fps} Hz[/cyan]")

    # robot motion
    root_pos = np.asarray(robot_motion["root_pos"], dtype=float)       # (T,3)
    root_rot_xyzw = np.asarray(robot_motion["root_rot"], dtype=float)  # (T,4) xyzw
    dof_pos = np.asarray(robot_motion["dof_pos"], dtype=float)         # (T,ndof)

    T = root_pos.shape[0]
    if root_rot_xyzw.shape[0] != T or dof_pos.shape[0] != T:
        raise ValueError("robot_motion 中 root_pos / root_rot / dof_pos 的帧数不一致。")

    if args.max_frames > 0:
        T = min(T, args.max_frames)

    print(f"[cyan]回放帧数: {T}[/cyan]")

    # viewer
    viewer = RobotMotionViewer_warn(
        robot_type=args.robot,
        motion_fps=motion_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path if args.record_video else None,
    )

    collision_log = []  # list[list[contact_dict]] per frame

    pbar = tqdm(total=T, desc="Replay motion", ncols=100)
    for i in range(T):
        rr_xyzw = root_rot_xyzw[i]
        rr_wxyz = np.array([rr_xyzw[3], rr_xyzw[0], rr_xyzw[1], rr_xyzw[2]], dtype=float)

        if args.show_human and human_motion is not None:
            human_dict = reconstruct_human_dict_for_frame(human_motion, i)
        else:
            human_dict = None

        # ★ 关键：让 viewer.step 自己检测并高亮碰撞，同时把 contacts 返回用于统计
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

        if args.print_collision_frames and frame_contacts:
            print(f"[Collision] frame {i}: {len(frame_contacts)} contacts")
            for c in frame_contacts:
                print(
                    f"  {c['body1']} <-> {c['body2']} | "
                    f"geom: {c['geom1']} vs {c['geom2']} | dist={c['dist']:.6f}"
                )

        pbar.update(1)

    pbar.close()
    viewer.close()

    # ------------ 碰撞汇总 ------------
    total_frames = len(collision_log)
    frames_with_collision = sum(1 for fc in collision_log if len(fc) > 0)
    total_contacts = sum(len(fc) for fc in collision_log)
    mean_contacts_per_frame = total_contacts / max(1, total_frames)

    print(f"\n[Collision Summary] {frames_with_collision}/{total_frames} frames have self-collision.")
    print(f"[Collision Detail] total contacts = {total_contacts}, "
          f"mean contacts/frame = {mean_contacts_per_frame:.3f}")

    # contact-weighted statistics
    all_dists = [c["dist"] for fc in collision_log for c in fc]
    if all_dists:
        all_dists = np.asarray(all_dists, dtype=float)
        mean_dist = float(np.mean(all_dists))
        min_dist = float(np.min(all_dists))
        max_dist = float(np.max(all_dists))

        penetrations = (-all_dists[all_dists < 0.0])  # convert to positive depth
        mean_penetration = float(np.mean(penetrations)) if penetrations.size > 0 else 0.0
        frac_penetrating = float(penetrations.size / all_dists.size)
    else:
        mean_dist = min_dist = max_dist = 0.0
        mean_penetration = 0.0
        frac_penetrating = 0.0

    print(f"[Collision Dist] mean(dist) = {mean_dist:.6f} m | "
          f"min(dist) = {min_dist:.6f} m | max(dist) = {max_dist:.6f} m")
    print(f"[Collision Penetration] mean(penetration) = {mean_penetration:.6f} m | "
          f"penetrating fraction = {frac_penetrating:.3%}")

    # 可选：按“碰撞帧”均值（每帧等权）
    per_frame_mean = [np.mean([c["dist"] for c in fc]) for fc in collision_log if fc]
    if per_frame_mean:
        print(f"[Collision Dist Per-Collision-Frame] mean = {float(np.mean(per_frame_mean)):.6f} m "
              f"(each collision frame equally weighted)")


if __name__ == "__main__":
    main()

"""
示例：
python scripts/replay_pkl_with_collisions_warn.py \
  --pkl_file /home/nubot/GMR_old/retargeting_result/g1_test_gmr/dance2_subject1_c.pkl \
  --robot unitree_g1 \
  --rate_limit 
  
  
  \
  --show_human \
  --print_collision_frames
"""
