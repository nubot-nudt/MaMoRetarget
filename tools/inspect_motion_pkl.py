import argparse
import pickle
from pathlib import Path

import numpy as np


def describe_array(name, arr):
    """小工具：打印 ndarray 的 shape / dtype"""
    if arr is None:
        print(f"  {name}: None")
        return
    arr = np.asarray(arr)
    print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")


def describe_motion_block(block_name, motion):
    """
    打印一个 motion 块（robot_motion 或 human_motion）的结构：
    - 有哪些 key
    - 各个数组的形状
    - link_body_list 里的关节名
    """
    if motion is None:
        print(f"=== {block_name}: None ===")
        return

    print(f"\n=== {block_name} ===")
    print("keys:", list(motion.keys()))

    # 常见字段
    for key in ["fps", "root_pos", "root_rot", "dof_pos", "local_body_pos"]:
        if key in motion:
            if key == "fps":
                print(f"  fps: {motion[key]}")
            else:
                describe_array(key, motion[key])

    # 关节名字列表
    if "link_body_list" in motion:
        names = motion["link_body_list"] #dof_names
        if names is None:
            print("  link_body_list: None")
        else:
            print(f"  link_body_list (len={len(names)}):")
            for i, n in enumerate(names):
                print(f"    [{i:02d}] {n}")
    else:
        print("  link_body_list: (not found)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, required=True, help="要查看的 pkl 文件路径")
    args = parser.parse_args()

    pkl_path = Path(args.pkl)
    print(f"[Info] Loading: {pkl_path}")

    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    print("顶层类型:", type(data))
    if isinstance(data, dict):
        print("顶层 keys:", list(data.keys()))

    # 尝试按 GMR 现在的格式：
    #   {"robot_motion": {...}, "human_motion": {...}}
    if isinstance(data, dict) and "robot_motion" in data and "human_motion" in data:
        describe_motion_block("robot_motion", data["robot_motion"])
        describe_motion_block("human_motion", data["human_motion"])
    else:
        # 老格式：可能直接就是一个 motion dict
        print("\n[Warn] 顶层不是 {robot_motion, human_motion} 这种结构，"
              "尝试直接当作一个 motion block 打印：")
        describe_motion_block("motion", data)


if __name__ == "__main__":
    main()


"""
python tools/inspect_motion_pkl.py \
  --pkl net/dataset/val/output/aiming2_subject5_refine.pkl
"""