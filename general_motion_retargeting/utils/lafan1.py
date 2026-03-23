import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def load_lafan1_file(bvh_file):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    data = read_bvh(bvh_file)

    #print("--------------------------------")
    #print(data.bones)
    #算出每一帧每个关节的全局位姿
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)
    
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100  # cm to m
            result[bone] = (position, orientation)

        # Add modified foot pose
        result["LeftFootMod"] = (result["LeftFoot"][0], result["LeftToe"][1])
        result["RightFootMod"] = (result["RightFoot"][0], result["RightToe"][1])
        
        frames.append(result)
    
    # === 查看 frames 里有哪些键 ===
    # print(frames[0]['Hips'])
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
        # 预期：up.z > 0（Z朝上）；forward 大致落在 X/Y 平面且方向和你工程里“前”一致
        #[CHECK] up.z=0.687  forward=[-0.038 -0.994 -0.104]
    #print("frames:",frames)
    human_height = result["Head"][0][2] - min(result["LeftFootMod"][0][2], result["RightFootMod"][0][2])
    # human_height = human_height + 0.2  # cm to m
    human_height = 1.75  # cm to m

    return frames, human_height


