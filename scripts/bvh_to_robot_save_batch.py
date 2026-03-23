import argparse
import pathlib
import time
import os
import traceback
import numpy as np
import mujoco as mj
from rich import print
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_lafan1_file

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


def _resolve_video_path(video_path: str, bvh_stem: str) -> str:
    """
    批处理时，为每个 bvh 生成一个独立视频路径：
    - 若 video_path 包含 "{stem}"：按模板替换
    - 否则：
        - 若以 ".mp4" 结尾：使用同目录，文件名改为 "{stem}.mp4"
        - 否则视为目录：输出到 "<video_path>/{stem}.mp4"
    """
    if "{stem}" in video_path:
        return video_path.format(stem=bvh_stem)

    vp = pathlib.Path(video_path)
    if vp.suffix.lower() == ".mp4":
        out_dir = vp.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / f"{bvh_stem}.mp4")

    out_dir = vp
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{bvh_stem}.mp4")


def process_one_bvh(
    bvh_path: pathlib.Path,
    robot: str,
    motion_fps: int,
    rate_limit: bool,
    record_video: bool,
    video_path: str | None,
    save_path: str | None,
    transparent_robot: float = 0.0,
    save_collision_log: bool = False,
) -> None:
    """
    处理单个 BVH，并（可选）保存 pkl。
    保存格式与原脚本一致：{"robot_motion": ..., "human_motion": ...}，可选附带 "collision_log"。
    """
    # Load BVH
    lafan1_data_frames, actual_human_height = load_lafan1_file(str(bvh_path))

    # Initialize retargeter
    retargeter = GMR(
        src_human="bvh",
        tgt_robot=robot,
        actual_human_height=actual_human_height,
    )

    # Viewer
    robot_motion_viewer = RobotMotionViewer(
        robot_type=robot,
        motion_fps=motion_fps,
        transparent_robot=transparent_robot,
        record_video=record_video,
        video_path=(video_path or "videos/example.mp4"),
    )

    # Prepare save buffers
    qpos_list = []
    human_pos_list = []
    human_rot_list = []
    body_names = None

    robot_dof_names = None
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        robot_dof_names = _build_robot_dof_names(robot_motion_viewer.model)

    # FPS measurement
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    print(f"\n[Process] {bvh_path.name} | frames={len(lafan1_data_frames)} | mocap_fps={motion_fps}")
    pbar = tqdm(total=len(lafan1_data_frames), desc=f"Retargeting ({bvh_path.stem})")

    collision_log = []

    i = 0
    while i < len(lafan1_data_frames):
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        pbar.update(1)

        smplx_data = lafan1_data_frames[i]
        qpos = retargeter.retarget(smplx_data)

        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=rate_limit,
        )

        # ---- collision stats (robot self-collision only) ----
        model = robot_motion_viewer.model
        data = robot_motion_viewer.data
        frame_contacts = []

        for ci in range(data.ncon):
            contact = data.contact[ci]
            g1 = contact.geom1
            g2 = contact.geom2
            if g1 < 0 or g2 < 0:
                continue

            b1 = model.geom_bodyid[g1]
            b2 = model.geom_bodyid[g2]
            body1_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b1)
            body2_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b2)
            if body1_name == "world" or body2_name == "world":
                continue

            geom1_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g1)
            geom2_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g2)

            frame_contacts.append(
                {
                    "geom1": geom1_name,
                    "geom2": geom2_name,
                    "body1": body1_name,
                    "body2": body2_name,
                    "dist": float(contact.dist),
                    "normal": contact.frame[0:3].copy(),
                }
            )

        collision_log.append(frame_contacts)

        if frame_contacts:
            print(f"[Collision] frame {i}: {len(frame_contacts)} contacts")
            for c in frame_contacts:
                print(
                    f"  {c['body1']} <-> {c['body2']} | "
                    f"geom: {c['geom1']} vs {c['geom2']} | dist={c['dist']:.4f}"
                )

        # ---- cache for saving ----
        if save_path is not None:
            qpos_list.append(qpos)

            if body_names is None:
                body_names = build_human_target_list(retargeter)

                # 强校验：确保这些名字确实都在 scaled_human_data 里，否则说明配置/数据不一致
                missing = [n for n in body_names if n not in retargeter.scaled_human_data]
                if missing:
                    raise KeyError(
                        f"[human_motion] ik_match_table joints missing in scaled_human_data: {missing}\n"
                        f"available keys: {list(retargeter.scaled_human_data.keys())}"
            )


            pos_frame = np.stack([retargeter.scaled_human_data[name][0] for name in body_names], axis=0)
            rot_frame_wxyz = np.stack([retargeter.scaled_human_data[name][1] for name in body_names], axis=0)
            rot_frame_xyzw = rot_frame_wxyz[:, [1, 2, 3, 0]]

            human_pos_list.append(pos_frame)
            human_rot_list.append(rot_frame_xyzw)

        i += 1

    pbar.close()

    # ---- save pkl ----
    if save_path is not None:
        import pickle

        root_pos = np.array([q[:3] for q in qpos_list])
        root_rot = np.array([q[3:7][[1, 2, 3, 0]] for q in qpos_list])  # wxyz -> xyzw
        dof_pos = np.array([q[7:] for q in qpos_list])

        if robot_dof_names is not None and dof_pos.shape[1] != len(robot_dof_names):
            print(
                f"[Warn] dof_pos dim={dof_pos.shape[1]} != len(dof_names)={len(robot_dof_names)}. "
                "可能存在 ball/free joint 或模型 qpos 布局差异，请检查。"
            )

        robot_motion = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "dof_names": robot_dof_names,
            "local_body_pos": None,
            "link_body_list": None,
        }

        if len(human_pos_list) > 0:
            human_pos = np.stack(human_pos_list, axis=0)  # (T, N, 3)
            human_rot = np.stack(human_rot_list, axis=0)  # (T, N, 4) xyzw

            human_root_name = retargeter.human_root_name
            if human_root_name not in body_names:
                raise ValueError(
                    f"human_root_name={human_root_name} 不在 body_names={body_names} 里，请检查 scaled_human_data 的 key。"
                )
            root_idx = body_names.index(human_root_name)

            human_root_pos = human_pos[:, root_idx]
            human_root_rot = human_rot[:, root_idx]
            human_local_body_pos = human_pos - human_root_pos[:, None, :]

            human_motion = {
                "fps": motion_fps,
                "root_pos": human_root_pos,
                "root_rot": human_root_rot,
                "dof_pos": None,
                "local_body_pos": human_local_body_pos,
                "link_body_list": body_names,
            }
        else:
            human_motion = None

        save_data = {"robot_motion": robot_motion, "human_motion": human_motion}
        if save_collision_log:
            save_data["collision_log"] = collision_log

        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)

        print(f"[Save] {save_path}")

    # ---- collision summary ----
    total_frames = len(collision_log)
    frames_with_collision = sum(1 for fc in collision_log if len(fc) > 0)
    print(f"[Collision Summary] {frames_with_collision}/{total_frames} frames have self-collision.")

    robot_motion_viewer.close()


def main():
    parser = argparse.ArgumentParser()

    # 单文件 / 文件夹（二选一）
    parser.add_argument("--bvh_file", type=str, default=None, help="BVH motion file to load.")
    parser.add_argument("--bvh_dir", type=str, default=None, help="Directory containing BVH files to process.")

    # 文件夹模式参数
    parser.add_argument("--glob", type=str, default="*.bvh", help='Glob pattern under --bvh_dir (default: "*.bvh").')
    parser.add_argument("--recursive", action="store_true", help="Recursively search BVH files under --bvh_dir.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for pkls when using --bvh_dir.")
    parser.add_argument("--suffix", type=str, default="_c", help='Output filename suffix (default: "_c").')
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop immediately if any BVH fails.")

    # 兼容原脚本：单文件模式的 save_path
    parser.add_argument("--save_path", default=None, help="Path to save the robot motion (single-file mode).")

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

    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
        help='Single-file: exact mp4 path. Batch: dir / mp4 / template (supports "{stem}").',
    )

    parser.add_argument("--rate_limit", action="store_true", default=False)
    parser.add_argument("--motion_fps", type=int, default=30)

    parser.add_argument(
        "--save_collision_log",
        action="store_true",
        default=False,
        help="If set, include collision_log into the saved pkl (may affect downstream scripts).",
    )

    args = parser.parse_args()

    if (args.bvh_file is None) == (args.bvh_dir is None):
        parser.error("You must specify exactly ONE of --bvh_file or --bvh_dir.")

    # -------- single-file mode --------
    if args.bvh_file is not None:
        bvh_path = pathlib.Path(args.bvh_file)
        if not bvh_path.exists():
            raise FileNotFoundError(f"BVH not found: {bvh_path}")

        process_one_bvh(
            bvh_path=bvh_path,
            robot=args.robot,
            motion_fps=args.motion_fps,
            rate_limit=args.rate_limit,
            record_video=args.record_video,
            video_path=(args.video_path if args.record_video else None),
            save_path=args.save_path,
            save_collision_log=args.save_collision_log,
        )
        return

    # -------- batch mode --------
    bvh_dir = pathlib.Path(args.bvh_dir)
    if not bvh_dir.exists():
        raise FileNotFoundError(f"BVH dir not found: {bvh_dir}")

    if args.out_dir is None:
        raise ValueError("Batch mode requires --out_dir (where pkls will be saved).")

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.recursive:
        bvh_files = sorted(bvh_dir.rglob(args.glob))
    else:
        bvh_files = sorted(bvh_dir.glob(args.glob))

    bvh_files = [p for p in bvh_files if p.is_file()]
    if len(bvh_files) == 0:
        raise ValueError(f"No BVH files found under {bvh_dir} with glob={args.glob} (recursive={args.recursive}).")

    print(f"[Batch] Found {len(bvh_files)} BVH files under {bvh_dir}")

    ok, fail = 0, 0
    for idx, bvh_path in enumerate(bvh_files, start=1):
        stem = bvh_path.stem
        save_path = str(out_dir / f"{stem}{args.suffix}.pkl")

        if (not args.overwrite) and os.path.exists(save_path):
            print(f"[Skip] ({idx}/{len(bvh_files)}) {bvh_path.name} -> exists: {save_path}")
            continue

        vpath = None
        if args.record_video:
            vpath = _resolve_video_path(args.video_path, stem)

        print(f"\n========== ({idx}/{len(bvh_files)}) {bvh_path} ==========")
        try:
            process_one_bvh(
                bvh_path=bvh_path,
                robot=args.robot,
                motion_fps=args.motion_fps,
                rate_limit=args.rate_limit,
                record_video=args.record_video,
                video_path=vpath,
                save_path=save_path,
                save_collision_log=args.save_collision_log,
            )
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[Error] Failed on {bvh_path}: {e}")
            traceback.print_exc()
            if args.fail_fast:
                raise

    print(f"\n[Batch Done] ok={ok}, fail={fail}, out_dir={out_dir}")


if __name__ == "__main__":
    main()


"""
python scripts/bvh_to_robot_save_batch.py \
  --bvh_dir lafan1/bvh_test \
  --glob "*.bvh" \
  --out_dir retargeting_result/baseline/PM/n1 \
  --robot fourier_n1 \
  --rate_limit

python scripts/bvh_to_robot_save_batch.py \
  --bvh_file lafan1/multipleActions1_subject4.bvh \
  --robot unitree_g1 \
  --save_path retargeting_result/lafan1_g1_save/multipleActions1_subject4_c.pkl \
  --rate_limit
"""