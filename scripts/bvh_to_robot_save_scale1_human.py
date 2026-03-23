import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_lafan1_file


# ---------------- Quaternion helpers (wxyz) ----------------
def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out

def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)

def quat_inv_wxyz(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q)
    conj = quat_conj_wxyz(q)
    norm2 = np.sum(q*q, axis=-1, keepdims=True)
    return conj / np.maximum(norm2, eps)

def wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q_wxyz = np.asarray(q_wxyz)
    return q_wxyz[..., [1, 2, 3, 0]]

def xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(q_xyzw)
    return q_xyzw[..., [3, 0, 1, 2]]


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


def extract_targets_from_scaled_human_data(scaled_human_data: dict, body_names: list):
    """
    从 retargeter.scaled_human_data（wxyz）按 body_names 顺序抽取 (pos, quat_xyzw)
    返回:
      pos: (N,3)
      rot_xyzw: (N,4)
    """
    pos = []
    rot_xyzw = []
    for n in body_names:
        if n not in scaled_human_data:
            raise KeyError(f"scaled_human_data missing key '{n}'. Available keys: {list(scaled_human_data.keys())[:10]} ...")
        p, q_wxyz = scaled_human_data[n]
        pos.append(np.asarray(p, dtype=np.float32))
        rot_xyzw.append(wxyz_to_xyzw(np.asarray(q_wxyz, dtype=np.float32)))
    return np.stack(pos, axis=0), np.stack(rot_xyzw, axis=0)


def compute_root_relative(pos_global: np.ndarray, rot_global_xyzw: np.ndarray, root_idx: int):
    """
    pos_global: (T,N,3)  world
    rot_global_xyzw: (T,N,4) world
    返回:
      root_pos (T,3), root_rot_xyzw (T,4),
      local_pos (T,N,3), local_rot_xyzw (T,N,4)  (root-relative)
    """
    root_pos = pos_global[:, root_idx]
    root_rot = rot_global_xyzw[:, root_idx]

    local_pos = pos_global - root_pos[:, None, :]

    root_wxyz = xyzw_to_wxyz(root_rot)              # (T,4)
    joint_wxyz = xyzw_to_wxyz(rot_global_xyzw)      # (T,N,4)
    root_inv = quat_inv_wxyz(root_wxyz)[:, None, :] # (T,1,4)
    local_wxyz = quat_mul_wxyz(root_inv, joint_wxyz)
    local_rot_xyzw = wxyz_to_xyzw(local_wxyz).astype(np.float32)

    return root_pos.astype(np.float32), root_rot.astype(np.float32), local_pos.astype(np.float32), local_rot_xyzw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", required=True, type=str)
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy",
                 "fourier_n1", "engineai_pm01", "zlite"],
        default="unitree_g1",
    )
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--save_human_gmr", action="store_true",
                        help="额外保存一份 human_motion_gmr（正常GMR scaled_human_data）用于对比/消融")
    parser.add_argument("--offset_to_ground", action="store_true", default=True,
                        help="是否执行 offset_human_data_to_ground（默认True，跟你原流程一致）")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # Load BVH frames
    lafan1_data_frames, actual_human_height = load_lafan1_file(args.bvh_file)
    motion_fps = 30

    # retargeter_label: 正常GMR（生成 robot_motion teacher）
    retargeter_label = GMR(src_human="bvh", tgt_robot=args.robot, actual_human_height=actual_human_height)

    # retargeter_scale1: 同一套json（含 rot_offset/pos_offset/ground），但 scale 置1
    retargeter_scale1 = GMR(src_human="bvh", tgt_robot=args.robot, actual_human_height=actual_human_height)
    retargeter_scale1.human_scale_table = {k: 1.0 for k in retargeter_scale1.human_scale_table.keys()}

    # human target joints 顺序（固定）
    body_names = build_human_target_list(retargeter_label)
    root_name = retargeter_label.human_root_name
    root_idx = body_names.index(root_name)

    # Buffers
    qpos_list = []

    # scale=1 aligned (仍保留 rot_offset/pos_offset/ground)
    h1_pos_list = []
    h1_rot_list = []

    # optional: gmr scaled
    hg_pos_list = []
    hg_rot_list = []

    for frame in tqdm(lafan1_data_frames, desc="Retarget + Save(scale=1 aligned)"):
        # 1) teacher qpos（会更新 retargeter_label.scaled_human_data）
        qpos = retargeter_label.retarget(frame, offset_to_ground=args.offset_to_ground)
        qpos_list.append(qpos)

        # 2) scale=1 aligned human targets（仅update_targets，不解IK）
        retargeter_scale1.update_targets(frame, offset_to_ground=args.offset_to_ground)
        pos1, rot1 = extract_targets_from_scaled_human_data(retargeter_scale1.scaled_human_data, body_names)
        h1_pos_list.append(pos1)
        h1_rot_list.append(rot1)

        # 3) 可选保存正常GMR的 human targets（用于debug/消融）
        if args.save_human_gmr:
            posg, rotg = extract_targets_from_scaled_human_data(retargeter_label.scaled_human_data, body_names)
            hg_pos_list.append(posg)
            hg_rot_list.append(rotg)

    # ---------------- robot_motion ----------------
    qpos_all = np.stack(qpos_list, axis=0).astype(np.float32)  # (T, 7+DoF)
    robot_motion = {
        "fps": motion_fps,
        "root_pos": qpos_all[:, :3],
        "root_rot": qpos_all[:, 3:7][:, [1, 2, 3, 0]],  # wxyz->xyzw
        "dof_pos": qpos_all[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }

    # ---------------- human_motion_scale1 ----------------
    h1_pos = np.stack(h1_pos_list, axis=0).astype(np.float32)  # (T,N,3)
    h1_rot = np.stack(h1_rot_list, axis=0).astype(np.float32)  # (T,N,4) xyzw

    h1_root_pos, h1_root_rot, h1_local_pos, h1_local_rot = compute_root_relative(h1_pos, h1_rot, root_idx)

    human_motion_scale1 = {
        "fps": motion_fps,
        "root_pos": h1_root_pos,
        "root_rot": h1_root_rot,
        "global_body_pos": h1_pos,         # (T,N,3) world，但已应用 rot_offset/pos_offset/ground
        "global_body_rot": h1_rot,         # (T,N,4) world，但已应用 rot_offset（坐标系对齐）
        "local_body_pos": h1_local_pos,    # (T,N,3) root-translation removed
        "local_body_rot": h1_local_rot,    # (T,N,4) root-relative rotations
        "link_body_list": body_names,
        "note": "scale=1 (human_scale_table all ones), but still applies json rot_offset/pos_offset/ground via update_targets()",
    }

    save_data = {
        "robot_motion": robot_motion,
        "human_motion_scale1": human_motion_scale1,
    }

    # ---------------- optional human_motion_gmr ----------------
    if args.save_human_gmr:
        hg_pos = np.stack(hg_pos_list, axis=0).astype(np.float32)
        hg_rot = np.stack(hg_rot_list, axis=0).astype(np.float32)
        hg_root_pos, hg_root_rot, hg_local_pos, hg_local_rot = compute_root_relative(hg_pos, hg_rot, root_idx)
        save_data["human_motion"] = {
            "fps": motion_fps,
            "root_pos": hg_root_pos,
            "root_rot": hg_root_rot,
            "global_body_pos": hg_pos,
            "global_body_rot": hg_rot,
            "local_body_pos": hg_local_pos,
            "local_body_rot": hg_local_rot,
            "link_body_list": body_names,
            "note": "normal GMR scaled_human_data (uses human_scale_table)",
        }

    with open(args.save_path, "wb") as f:
        pickle.dump(save_data, f)

    print(f"\n[OK] Saved to: {args.save_path}")
    print(f"  T={qpos_all.shape[0]}, N_targets={len(body_names)}, DoF={qpos_all.shape[1]-7}")
    print(f"  human targets order: {body_names}")


if __name__ == "__main__":
    main()


"""
python scripts/bvh_to_robot_save_scale1_human.py \
  --bvh_file lafan1/dance1_subject2.bvh \
  --robot unitree_g1 \
  --save_path net/dataset/all_input/dance1_subject2.pkl \
  --save_human_gmr
"""