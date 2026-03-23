import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_lafan1_file
from rich import print
from tqdm import tqdm
import os
import numpy as np
import mujoco as mj   # ← 新增：用于读取 contact / body 名字

def build_human_target_list(retargeter: GMR):
    """
    从 ik_match_table1/2 中提取 human target joints 的稳定顺序（按 table 遍历顺序，去重）。
    只取 pos_weight!=0 或 rot_weight!=0 的项（真正作为 target 的）。
    """
    names = []

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

def _qpos_size(jtype: int) -> int:
    # Mujoco joint qpos size: hinge/slide=1, ball=4, free=7
    if jtype == mj.mjtJoint.mjJNT_HINGE or jtype == mj.mjtJoint.mjJNT_SLIDE:
        return 1
    if jtype == mj.mjtJoint.mjJNT_BALL:
        return 4
    if jtype == mj.mjtJoint.mjJNT_FREE:
        return 7
    return 1

def _build_robot_dof_names(model: mj.MjModel) -> list[str]:
    # 按 qpos 地址排序，确保与 qpos 向量顺序一致；然后只取 qpos[7:] 的 dof 名字
    joint_infos = []
    for j in range(model.njnt):
        adr = int(model.jnt_qposadr[j])
        joint_infos.append((adr, j))
    joint_infos.sort(key=lambda x: x[0])

    dof_names: list[str] = []
    for adr, j in joint_infos:
        if adr < 7:  # 跳过 root free joint
            continue
        jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)
        jtype = int(model.jnt_type[j])
        sz = _qpos_size(jtype)

        if sz == 1:
            dof_names.append(jname)
        else:
            for k in range(sz):
                dof_names.append(f"{jname}[{k}]")
    return dof_names

if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="BVH motion file to load.",
        required=True,
        type=str,
    )
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1_29dof", "stanford_toddy", "fourier_n1", "engineai_pm01","zlite"],
        default="unitree_g1",
    )
        
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion.",
    )
    
    
    args = parser.parse_args()
    

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:  # Only create directory if it's not empty
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

        # 用于保存每一帧 “已经 scaled 并 set 为 target 的 human 数据”
        human_pos_list = []    # 每帧 (N_bodies, 3)
        human_rot_list = []    # 每帧 (N_bodies, 4)，统一存 xyzw
        body_names = None      # 记录关节名顺序

    
    # Load SMPLX trajectory
    lafan1_data_frames, actual_human_height = load_lafan1_file(args.bvh_file)
    #lafan1_data_frames每一帧全局下经过变换后的关节坐标
    
    # Initialize the retargeting system
    retargeter = GMR(
        src_human="bvh",
        tgt_robot=args.robot,
        actual_human_height=actual_human_height,
    )

    motion_fps = 30
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=args.record_video,
                                            video_path=args.video_path,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    # ---------- 新增：保存 robot dof 的名字顺序（与 qpos[7:] / dof_pos 完全对齐）----------
    robot_dof_names = None
    if args.save_path is not None:
        model = robot_motion_viewer.model

        def _qpos_size(jtype: int) -> int:
            # Mujoco joint qpos size: hinge/slide=1, ball=4, free=7
            if jtype == mj.mjtJoint.mjJNT_HINGE or jtype == mj.mjtJoint.mjJNT_SLIDE:
                return 1
            if jtype == mj.mjtJoint.mjJNT_BALL:
                return 4
            if jtype == mj.mjtJoint.mjJNT_FREE:
                return 7
            return 1

        # 按 qpos 地址排序，确保与 qpos 向量顺序一致
        joint_infos = []
        for j in range(model.njnt):
            adr = int(model.jnt_qposadr[j])
            joint_infos.append((adr, j))

        joint_infos.sort(key=lambda x: x[0])

        dof_names = []
        for adr, j in joint_infos:
            # root free joint 的 qposadr 通常从 0 开始，占 7 维；我们只要 qpos[7:]，所以跳过 adr < 7
            if adr < 7:
                continue
            jname = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)
            jtype = int(model.jnt_type[j])
            sz = _qpos_size(jtype)

            if sz == 1:
                dof_names.append(jname)
            else:
                # 多自由度 joint（ball/free）展开到每个 qpos 维度
                for k in range(sz):
                    dof_names.append(f"{jname}[{k}]")

        robot_dof_names = dof_names


    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=len(lafan1_data_frames), desc="Retargeting")
    
    # Start the viewer
    i = 0
    # 新增：用于记录每一帧的碰撞信息（list[ list[dict] ]）
    collision_log = []

    while i < len(lafan1_data_frames):
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        # Update task targets.
        smplx_data = lafan1_data_frames[i]

        # retarget
        qpos = retargeter.retarget(smplx_data)
        #print("-----------------------------")
        #print(qpos.shape)
        #print(qpos)
        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        # ================== 新增：统计这一帧的碰撞情况 ==================
        model = robot_motion_viewer.model
        data = robot_motion_viewer.data

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

        # 记录到全局 log 里
        collision_log.append(frame_contacts)

        # 如需调试，可打印这一帧发生的碰撞
        if frame_contacts:
            print(f"[Collision] frame {i}: {len(frame_contacts)} contacts")
            for c in frame_contacts:
                print(f"  {c['body1']} <-> {c['body2']} | "
                      f"geom: {c['geom1']} vs {c['geom2']} | dist={c['dist']:.4f}")

        # ================== 碰撞统计结束 ==================

        if args.save_path is not None:
            qpos_list.append(qpos)
            # print("-----------------------------")
            # print(retargeter.scaled_human_data)
            # print(sorted(retargeter.scaled_human_data))
            # 2) 保存当前帧的 human target 姿态
            # retargeter.retarget() 里已经调用了 update_targets，
            # 此时 retargeter.scaled_human_data 就是这一帧 “set 为 target 的 human 关节数据”
            if body_names is None:
                body_names = build_human_target_list(retargeter)

                # 强校验：确保这些名字确实都在 scaled_human_data 里，否则说明配置/数据不一致
                missing = [n for n in body_names if n not in retargeter.scaled_human_data]
                if missing:
                    raise KeyError(
                        f"[human_motion] ik_match_table joints missing in scaled_human_data: {missing}\n"
                        f"available keys: {list(retargeter.scaled_human_data.keys())}"
                    )

            # （T时刻）按 body_names 顺序取出 global pos / rot（这里是 world frame 下的 anchor）
            pos_frame = np.stack(
                [retargeter.scaled_human_data[name][0] for name in body_names],
                axis=0,  # (N_bodies, 3)
            )
            rot_frame_wxyz = np.stack(
                [retargeter.scaled_human_data[name][1] for name in body_names],
                axis=0,  # (N_bodies, 4)，当前是 wxyz
            )
            # 保存时统一成 xyzw（和 robot 的 root_rot 一致）
            rot_frame_xyzw = rot_frame_wxyz[:, [1, 2, 3, 0]]  # (N_bodies, 4)

            human_pos_list.append(pos_frame)
            human_rot_list.append(rot_frame_xyzw)

        i += 1

    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # save from wxyz to xyzw
        root_rot = np.array([qpos[3:7][[1,2,3,0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        local_body_pos = None
        robot_dof_names = _build_robot_dof_names(robot_motion_viewer.model)

        # sanity check：确保名字数量与 dof_pos 维度一致
        if robot_dof_names is not None and dof_pos.shape[1] != len(robot_dof_names):
            print(f"[Warn] dof_pos dim={dof_pos.shape[1]} != len(dof_names)={len(robot_dof_names)}. "
                  "可能存在 ball/free joint 或模型 qpos 布局差异，请检查。")

        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "dof_names": robot_dof_names,      # ✅ 新增：与 dof_pos 顺序对齐的关节名
            "local_body_pos": local_body_pos,
            "link_body_list": robot_dof_names,       # 你原来的字段（robot 目前是 None）
        }

        # motion_data = {
        #     "fps": motion_fps,
        #     "root_pos": root_pos,
        #     "root_rot": root_rot,
        #     "dof_pos": dof_pos,
        #     "local_body_pos": local_body_pos,
        #     "link_body_list": body_names,
        # }

        # with open(args.save_path, "wb") as f:
        #     pickle.dump(motion_data, f)
        # print(f"Saved to {args.save_path}")
        # ---------- human：根据缓存的 pos/rot 构建 ----------
        if len(human_pos_list) > 0:
            human_pos = np.stack(human_pos_list, axis=0)  # (T, N_bodies, 3)
            human_rot = np.stack(human_rot_list, axis=0)  # (T, N_bodies, 4) xyzw


            # root 关节名来自 GMR 的 ik_config，比如 "Hips"
            human_root_name = retargeter.human_root_name
            if human_root_name not in body_names:
                raise ValueError(
                    f"human_root_name={human_root_name} 不在 body_names={body_names} 里，"
                    "请检查 scaled_human_data 的 key。"
                )
            
            root_idx = body_names.index(human_root_name)

            human_root_pos = human_pos[:, root_idx]   # (T, 3)
            human_root_rot = human_rot[:, root_idx]   # (T, 4) xyzw

            # 每个关节相对 root 的位移
            human_local_body_pos = human_pos - human_root_pos[:, None, :]  # (T, N_bodies, 3)

            human_motion = {
                "fps": motion_fps,
                "root_pos": human_root_pos,
                "root_rot": human_root_rot,
                "dof_pos": None,
                "local_body_pos": human_local_body_pos,
                "link_body_list": body_names,
            }
        else:
            # 没有任何 human target 被记录（比如你中途退出了）
            human_motion = None

        # ---------- 一起写进 pkl ----------
        save_data = {
            "robot_motion": motion_data,
            "human_motion": human_motion,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(save_data, f)

        print(f"[Save] robot_motion & human_motion 已保存到 {args.save_path}")
        
    # 简单汇总一下碰撞情况（可选）
    total_frames = len(collision_log)
    frames_with_collision = sum(1 for fc in collision_log if len(fc) > 0)
    print(f"\n[Collision Summary] {frames_with_collision}/{total_frames} frames have self-collision.")

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
    
    #python scripts/bvh_to_robot_save.py --bvh_file lafan1/multipleActions1_subject4.bvh --robot engineai_pm01 --save_path retargeting_result/n1/multipleActions1_subject4_c.pkl --rate_limit
    """
    python scripts/bvh_to_robot_save.py --bvh_file lafan1/aiming1_subject1.bvh --robot fourier_n1 --save_path retargeting_result/n1/aiming1_subject1_c.pkl --rate_limit
    """