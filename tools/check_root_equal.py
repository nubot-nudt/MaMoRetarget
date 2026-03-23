import argparse
import pickle
import numpy as np

def quat_angle_rad(q1, q2, eps=1e-12):
    """
    q1, q2: (...,4) xyzw
    返回每帧的旋转角差(弧度)，对 q 和 -q 等价(用 |dot|)。
    """
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    # normalize
    q1 = q1 / (np.linalg.norm(q1, axis=-1, keepdims=True) + eps)
    q2 = q2 / (np.linalg.norm(q2, axis=-1, keepdims=True) + eps)

    dot = np.sum(q1 * q2, axis=-1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)  # abs() makes it sign-invariant
    return 2.0 * np.arccos(dot)

def main(pkl_path: str, pos_atol=1e-6, ang_atol_rad=1e-4):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    robot = data["robot_motion"]
    human = data["human_motion"]
    if human is None:
        raise ValueError("human_motion is None (可能是中途退出或未记录到 human target).")

    r_pos = np.asarray(robot["root_pos"])
    r_rot = np.asarray(robot["root_rot"])  # xyzw
    h_pos = np.asarray(human["root_pos"])
    h_rot = np.asarray(human["root_rot"])  # xyzw

    print("robot root_pos shape:", r_pos.shape, "root_rot shape:", r_rot.shape)
    print("human root_pos shape:", h_pos.shape, "root_rot shape:", h_rot.shape)

    T = min(len(r_pos), len(h_pos))
    r_pos = r_pos[:T]; r_rot = r_rot[:T]
    h_pos = h_pos[:T]; h_rot = h_rot[:T]

    # --- position diff ---
    pos_diff = r_pos - h_pos
    pos_l2 = np.linalg.norm(pos_diff, axis=-1)
    print("\n[Position diff]")
    print("  max |dx|:", np.max(np.abs(pos_diff[:, 0])))
    print("  max |dy|:", np.max(np.abs(pos_diff[:, 1])))
    print("  max |dz|:", np.max(np.abs(pos_diff[:, 2])))
    print("  max L2:", np.max(pos_l2))
    print("  mean L2:", np.mean(pos_l2))

    # --- rotation diff (sign-invariant) ---
    ang = quat_angle_rad(r_rot, h_rot)
    print("\n[Rotation diff]")
    print("  max angle (deg):", np.max(ang) * 180 / np.pi)
    print("  mean angle (deg):", np.mean(ang) * 180 / np.pi)

    # --- "are they same?" criteria ---
    pos_same = np.allclose(r_pos, h_pos, atol=pos_atol, rtol=0)
    rot_same = np.all(ang < ang_atol_rad)

    print("\n[Equality check]")
    print(f"  pos allclose(atol={pos_atol}):", bool(pos_same))
    print(f"  rot angle < {ang_atol_rad} rad (~{ang_atol_rad*180/np.pi:.5f} deg):", bool(rot_same))

    # show first few frames
    print("\n[First 3 frames preview]")
    for i in range(min(3, T)):
        print(f"Frame {i}:")
        print("  robot pos:", r_pos[i], "human pos:", h_pos[i], "L2:", pos_l2[i])
        print("  robot rot(xyzw):", r_rot[i], "human rot(xyzw):", h_rot[i],
              "angle(deg):", ang[i] * 180 / np.pi)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="path to saved pkl produced by bvh_to_robot_save.py")
    ap.add_argument("--pos_atol", type=float, default=1e-6)
    ap.add_argument("--ang_atol_rad", type=float, default=1e-4)
    args = ap.parse_args()
    main(args.pkl, pos_atol=args.pos_atol, ang_atol_rad=args.ang_atol_rad)

"""
python tools/check_root_equal.py --pkl retargeting_result/lafan1_g1_save/aiming1_subject1_c.pkl
"""