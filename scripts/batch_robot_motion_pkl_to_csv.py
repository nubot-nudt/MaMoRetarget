import argparse
import os
import pickle
import numpy as np


def _extract_motion_dict(pkl_obj, motion_key="robot_motion"):
    # bvh_to_robot_save.py: {"robot_motion": {...}, "human_motion": {.../None}}
    if isinstance(pkl_obj, dict) and motion_key in pkl_obj and isinstance(pkl_obj[motion_key], dict):
        return pkl_obj[motion_key]
    # bvh_to_robot.py / 原 GMR: 直接就是 motion dict
    return pkl_obj


def motion_dict_to_csv(motion_data: dict, csv_path: str, target_fps: float = 30.0):
    dof_pos = motion_data.get("dof_pos", None)
    if dof_pos is None:
        raise ValueError(f"dof_pos is None in motion_data, keys={list(motion_data.keys())}")

    frame_rate = float(motion_data.get("fps", target_fps))
    root_pos = np.asarray(motion_data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(motion_data["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(dof_pos, dtype=np.float32)

    T = dof_pos.shape[0]
    if root_pos.shape[0] != T or root_rot.shape[0] != T:
        raise ValueError(
            f"Length mismatch: root_pos {root_pos.shape}, root_rot {root_rot.shape}, dof_pos {dof_pos.shape}"
        )

    motion = np.zeros((T, dof_pos.shape[1] + 7), dtype=np.float32)
    motion[:, :3] = root_pos
    motion[:, 3:7] = root_rot
    motion[:, 7:] = dof_pos

    if frame_rate > target_fps:
        downsample_factor = frame_rate / target_fps
        indices = np.arange(0, motion.shape[0], downsample_factor).astype(int)
        indices = np.unique(np.clip(indices, 0, motion.shape[0] - 1))
        old_len = motion.shape[0]
        motion = motion[indices]
        print(f"Downsampled from {old_len} to {motion.shape[0]} frames ({frame_rate} -> {target_fps} fps)")

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    np.savetxt(csv_path, motion, delimiter=",")
    return motion.shape


def convert_folder(folder: str, out_folder: str | None = None, motion_key: str = "robot_motion", target_fps: float = 30.0):
    if out_folder is None:
        out_folder = os.path.join(folder, "csv")
    os.makedirs(out_folder, exist_ok=True)

    pkl_files = sorted([f for f in os.listdir(folder) if f.endswith(".pkl")])
    for i, file in enumerate(pkl_files, 1):
        pkl_path = os.path.join(folder, file)
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

        motion_data = _extract_motion_dict(obj, motion_key=motion_key)
        csv_path = os.path.join(out_folder, file.replace(".pkl", ".csv"))
        shape = motion_dict_to_csv(motion_data, csv_path, target_fps=target_fps)

        fps_in = motion_data.get("fps", None)
        print(f"[{i}/{len(pkl_files)}] Saved {csv_path} | fps={fps_in} | shape={shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert robot motion pkl (from bvh_to_robot.py or bvh_to_robot_save.py) to CSV."
    )
    parser.add_argument("--folder", type=str, required=True, help="Folder containing .pkl files")
    parser.add_argument("--out_folder", type=str, default=None, help="Output folder (default: <folder>/csv)")
    parser.add_argument("--motion_key", type=str, default="robot_motion", help="Key to extract motion dict (default: robot_motion)")
    parser.add_argument("--target_fps", type=float, default=30.0, help="Downsample to this FPS if input FPS is higher")
    args = parser.parse_args()

    convert_folder(args.folder, out_folder=args.out_folder, motion_key=args.motion_key, target_fps=args.target_fps)

#python scripts/batch_robot_motion_pkl_to_csv.py --folder /home/nubot/GMR_old/retargeting_result/baseline/PHC/g1