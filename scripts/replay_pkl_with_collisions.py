#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回放 bvh_to_robot_save.py / refine_retargeting.py 生成的 pkl，
以和原来一致的方式用 RobotMotionViewer 可视化，并统计自碰撞信息。

注意：
- pkl 中保存的 root_rot 是 xyzw
- RobotMotionViewer.step 期望的 root_rot 是 wxyz
- 这里会在回放时做 xyzw -> wxyz 的转换
"""

import argparse
import pathlib
import time
import pickle

import numpy as np
import mujoco as mj
from rich import print
from tqdm import tqdm

from general_motion_retargeting import RobotMotionViewer


def load_motion_pkl(pkl_path):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pkl_file",
        type=str,
        required=True,
        help="bvh_to_robot_save.py 或 refine_retargeting.py 保存的 pkl 路径",
    )
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1",
            "unitree_g1_with_hands",
            "booster_t1",
            "stanford_toddy",
            "fourier_n1",
            "engineai_pm01",
            "zlite",
        ],
        default="unitree_g1",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
        help="是否录制视频（由 RobotMotionViewer 负责）",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="/home/nubot/GMR_old/videos/aiming1_subject1_minchange_hinge.mp4",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
        help="是否按 fps 控制播放速度（True=实时，False=尽快播放）",
    )
    parser.add_argument(
        "--show_human",
        action="store_true",
        default=False,
        help="是否在 viewer 中同时渲染 human（用 human_motion 重建，旋转近似）",
    )

    args = parser.parse_args()

    pkl_path = pathlib.Path(args.pkl_file)
    print(f"[cyan]加载 pkl:[/cyan] {pkl_path}")

    robot_motion, human_motion = load_motion_pkl(pkl_path)

    # 读取 fps
    fps_r = robot_motion.get("fps", None)
    fps_h = human_motion.get("fps", None) if human_motion is not None else None
    if fps_r is None and fps_h is None:
        raise ValueError("pkl 中 robot_motion 和 human_motion 都没有 'fps' 字段。")

    motion_fps = fps_r if fps_r is not None else fps_h
    print(f"[cyan]使用 fps = {motion_fps} Hz[/cyan]")

    root_pos = np.asarray(robot_motion["root_pos"], dtype=float)  # (T,3)
    root_rot_xyzw = np.asarray(robot_motion["root_rot"], dtype=float)  # (T,4) xyzw
    dof_pos = np.asarray(robot_motion["dof_pos"], dtype=float)    # (T,ndof)

    T = root_pos.shape[0]
    if not (root_rot_xyzw.shape[0] == T and dof_pos.shape[0] == T):
        raise ValueError("robot_motion 中 root_pos / root_rot / dof_pos 的帧数不一致。")

    print(f"[cyan]总帧数: {T}[/cyan]")

    # 初始化 RobotMotionViewer（和 bvh_to_robot_save.py 保持一致）
    viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=motion_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    # 碰撞统计
    collision_log = []

    pbar = tqdm(total=T, desc="Replay motion")

    for i in range(T):
        # 把保存的 root_rot (xyzw) 转回 wxyz 再喂给 viewer
        rr_xyzw = root_rot_xyzw[i]
        rr_wxyz = np.array([rr_xyzw[3], rr_xyzw[0], rr_xyzw[1], rr_xyzw[2]], dtype=float)

        # optional human
        if args.show_human and human_motion is not None:
            human_dict = reconstruct_human_dict_for_frame(human_motion, i)
        else:
            human_dict = None

        viewer.step(
            root_pos=root_pos[i],
            root_rot=rr_wxyz,      # ★ 正确：wxyz
            dof_pos=dof_pos[i],
            human_motion_data=human_dict,
            rate_limit=args.rate_limit,
        )

        # ======== 统计这一帧的自碰撞（与你之前代码一致） ========
        model = viewer.model
        data = viewer.data

        frame_contacts = []  # 本帧所有“机器人内部”碰撞（不含地面）

        for ci in range(data.ncon):  # 当前帧接触数量
            contact = data.contact[ci]
            g1 = contact.geom1
            g2 = contact.geom2

            # 有些 contact 可能是无效的（-1），先跳过
            if g1 < 0 or g2 < 0:
                continue

            # geom → body
            b1 = model.geom_bodyid[g1]
            b2 = model.geom_bodyid[g2]

            body1_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b1)
            body2_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b2)
            geom1_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g1)
            geom2_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g2)

            # 过滤掉与 world / 地面的接触，只保留“机器人自己各 link 之间”的碰撞
            if body1_name == "world" or body2_name == "world":
                continue

            frame_contacts.append({
                "geom1": geom1_name,
                "geom2": geom2_name,
                "body1": body1_name,
                "body2": body2_name,
                "dist": float(contact.dist),          # 接触间距（负值表示穿透）
                "normal": contact.frame[0:3].copy(), # 接触法向
            })

        collision_log.append(frame_contacts)

        if frame_contacts:
            print(f"[Collision] frame {i}: {len(frame_contacts)} contacts")
            for c in frame_contacts:
                print(
                    f"  {c['body1']} <-> {c['body2']} | "
                    f"geom: {c['geom1']} vs {c['geom2']} | dist={c['dist']:.4f}"
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
    


    # ======== 新增：所有碰撞 contact 的平均 dist ========
    all_dists = [c["dist"] for fc in collision_log for c in fc]
    if len(all_dists) > 0:
        mean_dist = float(np.mean(all_dists))     # 平均 signed distance（可能为负）
        min_dist = float(np.min(all_dists))       # 最深穿透（最负）
        max_dist = float(np.max(all_dists))       # 最大间距（可能接近 0 或略正）

        penetrations = [-d for d in all_dists if d < 0]  # 穿透深度（转为正数）
        mean_penetration = float(np.mean(penetrations)) if penetrations else 0.0
        frac_penetrating = len(penetrations) / len(all_dists)

    else:
        mean_dist = min_dist = max_dist = 0.0
        mean_penetration = 0.0
        frac_penetrating = 0.0
    

    print(f"[Collision Dist] mean(dist) = {mean_dist:.6f} m | "
          f"min(dist) = {min_dist:.6f} m | max(dist) = {max_dist:.6f} m")
    print(f"[Collision Penetration] mean(penetration) = {mean_penetration:.6f} m | "
          f"penetrating fraction = {frac_penetrating:.3%}")
    


if __name__ == "__main__":
    main()

"""
python scripts/replay_pkl_with_collisions.py \
  --pkl_file /home/nubot/GMR_old/retargeting_result/g1_test_gmr/dance2_subject1_c.pkl \
  --robot unitree_g1 \
  --rate_limit 
  
  
  \
  --show_human 
"""
