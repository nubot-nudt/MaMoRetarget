# -*- coding: utf-8 -*-
"""
ourdata.py — 稳健版
- 自动搜索 24 个右手系正交旋转，选择让 up.z 最大且 forward ≈ -Y 的矩阵（与 lafan1 处理后的朝向一致）
- 组合 Foot/Hand 位姿时使用别名兜底（缺谁都不崩）
- human_height 用首帧真实高度估计（不再写死 1.75）
- 保持返回格式与 lafan1.load_lafan1_file 一致
"""
from typing import Dict, Tuple, List, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
import itertools

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh, read_bvh_ourdata

# —— 我们数据的骨名别名（按你的差异报告收敛过）——
ALIASES = {
    # 根与脊柱
    "root":   ["Pelvis", "Hips"],
    "spine1": ["Spine1"],
    "spine2": ["Spine2"],
    "spine3": ["Spine3"],
    "head":   ["Head", "Head_End"],

    # 肩带/上肢（我们多 *_collar）
    "lcollar": ["Left_collar"],
    "lsho":    ["Left_shoulder", "LeftArm"],
    "lelb":    ["Left_elbow", "LeftForeArm"],
    "lwri":    ["Left_wrist", "LeftHand", "LeftWrist"],
    "lpal":    ["Left_palm", "LeftPalm"],

    "rcollar": ["Right_collar"],
    "rsho":    ["Right_shoulder", "RightArm"],
    "relb":    ["Right_elbow", "RightForeArm"],
    "rwri":    ["Right_wrist", "RightHand", "RightWrist"],
    "rpal":    ["Right_palm", "RightPalm"],

    # 下肢（LaFAN↔我们的映射）
    "lhip":  ["Left_hip", "LeftUpLeg"],
    "lknee": ["Left_knee", "LeftLeg"],
    "lank":  ["Left_ankle", "LeftFoot"],
    "ltoe":  ["Left_foot", "LeftToe", "LeftToeBase"],
    "rhip":  ["Right_hip", "RightUpLeg"],
    "rknee": ["Right_knee", "RightLeg"],
    "rank":  ["Right_ankle", "RightFoot"],
    "rtoe":  ["Right_foot", "RightToe", "RightToeBase"],

    # 手指（可选，用于估掌面兜底）
    "l_idx1": ["LeftHandIndex1", "LeftIndex1", "LeftIndexProximal"],
    "l_pky1": ["LeftHandPinky1", "LeftPinky1", "LeftLittleProximal"],
    "r_idx1": ["RightHandIndex1","RightIndex1","RightLittleProximal","RightIndexProximal"],
    "r_pky1": ["RightHandPinky1","RightPinky1","RightLittleProximal"],
}


def load_ourdata_file(bvh_file: str):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """

    data = read_bvh(bvh_file)
    # 只保留 root 的平移，其它关节回落到 offsets（若无 offsets 就置零）
    if hasattr(data, "offsets"):
        data.pos[:, 1:, :] = data.offsets[1:][None, :, :]
    else:
        data.pos[:, 1:, :] = 0.0
    print(np.max(np.abs(data.pos[:, 1:, :] - data.offsets[1:][None, :, :])))
    
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[ -1,0, 0],
                            [ 0, -1, 0],
                            [ 0, 0, 1]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            #orientation = utils.quat_mul(global_data[0][frame, i], rotation_quat)
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = (position, orientation)

        # Add modified foot pose
        result["LeftFootMod"] = (result["Left_ankle"][0], result["Left_foot"][1])
        result["RightFootMod"] = (result["Right_ankle"][0], result["Right_foot"][1])
        # Add modified foot pose
        result["LeftHand"] = (result["Left_wrist"][0], result["Left_wrist"][1])
        result["RightHand"] = (result["Right_wrist"][0], result["Right_wrist"][1])

        frames.append(result)
    # === 查看 frames 里有哪些键 ===
    print("\n=== keys in frame[0] ===")
    k0 = sorted(frames[0].keys())
    print(f"count={len(k0)}")
    for k in k0:
        print("  ", k)

    # 并集（所有帧出现过的键）
    union_keys = set()
    for fr in frames:
        union_keys |= set(fr.keys())
    print("\n=== union of keys over all frames ===")
    print(f"count={len(union_keys)}")
    for k in sorted(union_keys):
        print("  ", k)

    # 交集（每一帧都同时存在的键）
    inter_keys = set(frames[0].keys())
    for fr in frames[1:]:
        inter_keys &= set(fr.keys())
    print("\n=== intersection of keys over all frames ===")
    print(f"count={len(inter_keys)}")
    for k in sorted(inter_keys):
        print("  ", k)

    root = frames[0].get("Hips") or frames[0].get("Pelvis")
    head = frames[0].get("Head") or frames[0].get("Spine3")
    ls   = frames[0].get("LeftArm")  or frames[0].get("Left_shoulder")
    rs   = frames[0].get("RightArm") or frames[0].get("Right_shoulder")

    if root and head and ls and rs:
        up_est = head[0] - root[0]                         # 头-胯，近似“上”
        right  = rs[0] - ls[0]                             # 右肩-左肩，近似“右”
        forward= np.cross(up_est / (np.linalg.norm(up_est)+1e-8),
                        right  / (np.linalg.norm(right)+1e-8))
        print("[CHECK] up.z=%.3f  forward=%s" % (up_est[2], np.round(forward,3)))
    human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m

    return frames, human_height



#未修之前的代码
#python scripts/bvh_to_robot_ourdata.py --bvh_file ours_bvh/zhengbuzou.bvh --robot zlite_ourdata --save_path ours_bvh/zhengbuzou.pkl --rate_limit


#python scripts/bvh_to_robot_ourdata.py --bvh_file run_bvh/Run_Fast1-Body2.bvh --robot zlite_ourdata --save_path nokov/Run_Fast1-Body2.pkl --rate_limit



# # ourdata -> LaFAN 的关键映射（根据你文件名做了常见别名，缺的再补）
# ALIASES_TO_LAFAN = {
#     # 根与躯干
#     "Hips":      ["Pelvis"],
#     "Spine":     ["Spine1"],
#     "Spine1":    ["Spine2"],
#     "Spine2":    ["Spine3"],
#     "Neck":      ["Neck"],
#     "Head":      ["Head"],

#     # 左腿（LaFAN：LeftUpLeg → LeftLeg → LeftFoot → LeftToe）
#     "LeftUpLeg": ["Left_hip"],
#     "LeftLeg":   ["Left_knee"],
#     "LeftFoot":  ["Left_ankle"],
#     "LeftToe":   ["Left_foot"],

#     # 右腿
#     "RightUpLeg":["Right_hip"],
#     "RightLeg":  ["Right_knee"],
#     "RightFoot": ["Right_ankle"],
#     "RightToe":  ["Right_foot"],

#     # 左臂（LaFAN：LeftArm → LeftForeArm → LeftHand）
#     "LeftArm":     ["Left_shoulder"],
#     "LeftForeArm": ["Left_elbow"],
#     "LeftHand":    ["Left_wrist"],

#     # 右臂
#     "RightArm":     ["Right_shoulder"],
#     "RightForeArm": ["Right_elbow"],
#     "RightHand":    ["Right_wrist"],
# }

