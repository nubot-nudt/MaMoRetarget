#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
打印 G1 机器人 qpos / dof_pos 的布局：
- 根的位置 / 旋转
- 每个 dof_pos[i] 对应的 joint 名字
"""

import argparse
from pathlib import Path
import mujoco as mj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot_xml", type=str, required=True,
                        help="MuJoCo 机器人 xml 路径，例如 assets/unitree_g1/g1_mocap_29dof.xml")
    args = parser.parse_args()

    model = mj.MjModel.from_xml_path(args.robot_xml)

    print("=== qpos 布局 ===")
    if model.nq >= 7:
        print("qpos[0:3] : root_pos (x, y, z)")
        print("qpos[3:7] : root_quat (w, x, y, z)")
        print(f"qpos[7:{7+model.nv}] : dof_pos (共 {model.nv} 个 DoF)")
    else:
        print(f"nq={model.nq} < 7，说明 root 可能不是 free joint，请自行检查 xml。")

    print("\n=== dof_pos 每一维对应的 joint ===")
    for i in range(model.nv):
        jnt_id = model.dof_jntid[i]
        jnt_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jnt_id)
        print(f"dof_pos[{i:02d}] -> joint '{jnt_name}'")

    print("\n提示：在 pkl 里，qpos = [root_pos, root_quat(wxyz), dof_pos]；")
    print("你保存时把 root_quat 转成了 xyzw，dof_pos 顺序就是上面这一列。")


if __name__ == "__main__":
    main()
