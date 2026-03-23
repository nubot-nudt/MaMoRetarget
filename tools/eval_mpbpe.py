#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
评估 GMR 自身 retargeting 误差的脚本：
- 从一个 pkl 中读取 robot_motion（qpos）和 human_motion（IK target）
- 用 MuJoCo 对机器人做 FK，得到各个 body 的世界坐标
- 从 human_motion 里还原各个人体 anchor 的世界坐标
- 根据 bvh_to_g1.json 里的 human↔robot 对应关系，对应配对后计算：
    - Eᵍ-mpbpe（global）
    - E_mpbpe（root-relative）
"""

import argparse
import pickle
from pathlib import Path
import json

import numpy as np
import mujoco as mj


# ----------------------------
# 工具函数：从 pkl 还原机器人 qpos
# ----------------------------

def load_robot_qpos_from_motion_dict(motion):
    """
    motion: 形如 bvh_to_robot.py 里保存的 robot_motion：
        {
            "fps": ...,
            "root_pos": (T, 3),
            "root_rot": (T, 4),  # 保存时是 xyzw
            "dof_pos":  (T, ndof),
            ...
        }
    返回：
        qpos: (T, nq) = [root_pos(3), root_quat(wxyz 4), dof_pos(...)]
    """
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)  # (T, 3)
    root_rot_xyzw = np.asarray(motion["root_rot"], dtype=np.float64)  # (T, 4)
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)  # (T, ndof)

    T = root_pos.shape[0]
    assert root_rot_xyzw.shape[0] == T and dof_pos.shape[0] == T

    nq = 3 + 4 + dof_pos.shape[1]
    qpos = np.zeros((T, nq), dtype=np.float64)

    # 平移
    qpos[:, :3] = root_pos

    # 注意：保存时是 xyzw，MuJoCo qpos 里要求 wxyz
    qpos[:, 3:7] = root_rot_xyzw[:, [3, 0, 1, 2]]

    # 关节角
    qpos[:, 7:] = dof_pos

    return qpos


# ----------------------------
# 工具函数：FK 求机器人 body 的世界坐标
# ----------------------------

def fk_robot_bodies(model, qpos_seq, robot_body_names):
    """
    给定模型 + qpos 序列，计算若干 robot body 的世界坐标。

    输入：
        model: mj.MjModel
        qpos_seq: (T, nq)
        robot_body_names: list[str]，MuJoCo 里的 body 名，比如 ["pelvis", "torso_link", ...]
    返回：
        xpos: (T, J, 3)
    """
    data = mj.MjData(model)
    T = qpos_seq.shape[0]
    J = len(robot_body_names)

    body_ids = []
    for name in robot_body_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in model.")
        body_ids.append(bid)

    xpos = np.zeros((T, J, 3), dtype=np.float64)

    for t in range(T):
        data.qpos[:] = qpos_seq[t]
        mj.mj_forward(model, data)
        for j, bid in enumerate(body_ids):
            xpos[t, j] = data.xpos[bid]

    return xpos


# ----------------------------
# 从 human_motion 还原 human anchor 世界坐标
# ----------------------------

def load_human_global_pos_from_motion_dict(human_motion):
    """
    human_motion 结构（参照你现在 bvh_to_robot.py 的保存方式）：
        {
            "fps": ...,
            "root_pos": (T, 3),           # Hips anchor 的 global pos
            "root_rot": (T, 4),           # root rot (xyzw) 用不着也行
            "dof_pos": None,
            "local_body_pos": (T, N, 3),  # 每个关节相对 root 的位移
            "link_body_list": [N 个 human 关节名字，对应 scaled_human_data 的顺序],
        }
    返回：
        human_global_pos: (T, N, 3)  # 世界坐标下所有关节 anchor 的位置
        human_body_names: list[str]
    """
    root_pos = np.asarray(human_motion["root_pos"], dtype=np.float64)  # (T, 3)
    local_body_pos = np.asarray(human_motion["local_body_pos"], dtype=np.float64)  # (T, N, 3)

    T, N = local_body_pos.shape[:2]
    assert root_pos.shape[0] == T

    # global = root_pos + local
    human_global_pos = local_body_pos + root_pos[:, None, :]  # (T, N, 3)
    human_body_names = list(human_motion["link_body_list"])  # human 关节名字列表

    return human_global_pos, human_body_names


# ----------------------------
# mpbpe 指标
# ----------------------------

def compute_Eg_mpbpe(x_ref, x_pred):
    """
    Global mpbpe:
        Eᵍ-mpbpe = mean_t mean_j || p_ref(t,j) - p_pred(t,j) ||_2
    x_ref, x_pred: (T, J, 3)
    返回：单位 m
    """
    assert x_ref.shape == x_pred.shape
    diff = x_ref - x_pred  # (T, J, 3)
    dist = np.linalg.norm(diff, axis=-1)  # (T, J)
    return dist.mean()


def compute_E_mpbpe(x_ref, x_pred, root_index=0):
    """
    Root-relative mpbpe:
        对每一侧都减去自身 root 的位置。
    x_ref, x_pred: (T, J, 3)
    root_index: 在 J 里的哪个 index 是 root 对（通常 pelvis/Hips 对应的那一对）
    返回：单位 m
    """
    assert x_ref.shape == x_pred.shape

    ref_root = x_ref[:, root_index:root_index+1, :]   # (T,1,3)
    pred_root = x_pred[:, root_index:root_index+1, :]

    ref_rel = x_ref - ref_root
    pred_rel = x_pred - pred_root

    diff = ref_rel - pred_rel
    dist = np.linalg.norm(diff, axis=-1)
    return dist.mean()


# ----------------------------
# main：组合在一起
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, required=True,
                        help="包含 robot_motion 和 human_motion 的 pkl 路径（bvh_to_robot.py 保存的那个）")
    parser.add_argument("--robot_xml", type=str, required=True,
                        help="MuJoCo 机器人 xml 路径（例如 g1.xml）")
    parser.add_argument("--ik_config", type=str, required=True,
                        help="IK 配置 json（例如 bvh_to_g1.json）")
    parser.add_argument("--tables", type=str, default="both",
                        choices=["1", "2", "both"],
                        help="使用 ik_match_table1 / 2 / both 中的哪些关节（默认 both）")
    args = parser.parse_args()
    print("------------------------")
    # 1) 读取 pkl
    with Path(args.pkl).open("rb") as f:
        data = pickle.load(f)

    robot_motion = data["robot_motion"]
    human_motion = data["human_motion"]
    assert human_motion is not None, "pkl 中没有 human_motion，请确认 bvh_to_robot.py 已按我们改过的方式保存。"

    # 2) MuJoCo 模型
    model = mj.MjModel.from_xml_path(args.robot_xml)

    # 3) 机器人 qpos 序列
    qpos_robot = load_robot_qpos_from_motion_dict(robot_motion)  # (T, nq)

    # 4) 人体 anchor 世界坐标
    human_global_pos, human_body_names = load_human_global_pos_from_motion_dict(human_motion)  # (T, Nh, 3)
    T_h, Nh, _ = human_global_pos.shape

    # 5) 读取 ik_config：找到 human ↔ robot 的对应关系
    with Path(args.ik_config).open("r", encoding="utf-8") as f:
        ik_cfg = json.load(f)

    use_table1 = args.tables in ("1", "both") and ik_cfg.get("use_ik_match_table1", True)
    use_table2 = args.tables in ("2", "both") and ik_cfg.get("use_ik_match_table2", True)

    pairs = []  # (robot_body_name, human_body_name, pos_weight)
    seen = set()

    if use_table1:
        for robot_body_name, entry in ik_cfg["ik_match_table1"].items():
            human_name, pos_w, rot_w, pos_off, rot_off = entry
            # mpbpe 只关心位置，有 pos 权重的才纳入
            if pos_w >= 0:
                key = (robot_body_name, human_name)
                if key not in seen:
                    pairs.append((robot_body_name, human_name, pos_w))
                    seen.add(key)

    if use_table2:
        for robot_body_name, entry in ik_cfg["ik_match_table2"].items():
            human_name, pos_w, rot_w, pos_off, rot_off = entry
            if pos_w > 0:
                key = (robot_body_name, human_name)
                if key not in seen:
                    pairs.append((robot_body_name, human_name, pos_w))
                    seen.add(key)

    if not pairs:
        raise ValueError("在 ik_match_table 中没有找到 pos_weight > 0 的关节，无法计算 mpbpe。")

    # 6) 根据 pairs 构造 robot_body_names & human 对应 index
    robot_body_names = []
    human_indices = []

    for robot_body_name, human_name, pos_w in pairs:
        if human_name not in human_body_names:
            # 理论上 human_motion.link_body_list 就是 scaled_human_data 的 key，应当包含这些 human_name
            raise ValueError(f"Human 关节 {human_name} 不在 human_motion['link_body_list'] 中。")

        robot_body_names.append(robot_body_name)
        human_indices.append(human_body_names.index(human_name))

    J = len(robot_body_names)
    print("评估关节对（robot ↔ human）：")
    for j in range(J):
        print(f"  {j}: {robot_body_names[j]}  <->  {human_body_names[human_indices[j]]}")

    # 7) 机器人这边：FK 得到每个 robot_body 的世界坐标
    #    时间长度用 min(T_robot, T_human) 对齐
    T_robot = qpos_robot.shape[0]
    T = min(T_robot, T_h)  # 保守一点，截到最短
    qpos_robot = qpos_robot[:T]
    human_global_pos = human_global_pos[:T]

    xpos_robot_all = fk_robot_bodies(model, qpos_robot, robot_body_names)  # (T, J, 3)

    # 8) 人体这边：从 human_global_pos 取出对应的 human joints
    xpos_human = np.zeros((T, J, 3), dtype=np.float64)
    for j, idx_h in enumerate(human_indices):
        xpos_human[:, j, :] = human_global_pos[:, idx_h, :]

    # 9) 确定 root 在 pairs 中的 index（一般 pelvis ↔ Hips）
    robot_root_name = ik_cfg["robot_root_name"]   # "pelvis"
    human_root_name = ik_cfg["human_root_name"]   # "Hips"

    root_index = None
    for j, (rb, hn, _) in enumerate(pairs):
        if rb == robot_root_name and hn == human_root_name:
            root_index = j
            break

    if root_index is None:
        # 找不到就默认 0，但建议你检查一下输出
        print("[Warning] 在 pairs 中没有找到 (robot_root_name, human_root_name) 这对，默认用第一个 pair 作为 root。")
        root_index = 0

    print(f"Root pair: index={root_index}, "
          f"robot={robot_body_names[root_index]}, human={human_body_names[human_indices[root_index]]}")

    # 10) 计算误差的 Min / Median / Mean / Max（单位：m）
    # ---------- Global 版 ----------
    # 每个元素是 || p_human(t,j) - p_robot(t,j) ||
    diff_global = xpos_human - xpos_robot_all          # (T, J, 3)
    dist_global = np.linalg.norm(diff_global, axis=-1) # (T, J)
    dist_global_flat = dist_global.reshape(-1)         # (T*J,)

    Eg_min_m   = float(dist_global_flat.min())
    Eg_med_m   = float(np.median(dist_global_flat))
    Eg_mean_m  = float(dist_global_flat.mean())
    Eg_max_m   = float(dist_global_flat.max())

    # ---------- Root-relative 版 ----------
    # 先对 each side 减掉各自 root 的位置
    ref_root  = xpos_human[:, root_index:root_index+1, :]      # (T, 1, 3)
    pred_root = xpos_robot_all[:, root_index:root_index+1, :]  # (T, 1, 3)

    ref_rel  = xpos_human      - ref_root    # (T, J, 3)
    pred_rel = xpos_robot_all  - pred_root   # (T, J, 3)

    diff_rel  = ref_rel - pred_rel
    dist_rel  = np.linalg.norm(diff_rel, axis=-1)   # (T, J)
    dist_rel_flat = dist_rel.reshape(-1)            # (T*J,)

    E_min_m   = float(dist_rel_flat.min())
    E_med_m   = float(np.median(dist_rel_flat))
    E_mean_m  = float(dist_rel_flat.mean())
    E_max_m   = float(dist_rel_flat.max())

    # 换算成 mm
    Eg_min_mm, Eg_med_mm, Eg_mean_mm, Eg_max_mm = [x * 1000.0 for x in (Eg_min_m, Eg_med_m, Eg_mean_m, Eg_max_m)]
    E_min_mm,  E_med_mm,  E_mean_mm,  E_max_mm  = [x * 1000.0 for x in (E_min_m,  E_med_m,  E_mean_m,  E_max_m)]

    print("======================================")
    print(f"使用 ik_match_table: {args.tables}")
    print("--------------------------------------")
    print("Eᵍ-mpbpe (Global, 单位 mm):")
    print(f"  min   = {Eg_min_mm:.3f}")
    print(f"  median= {Eg_med_mm:.3f}")
    print(f"  mean  = {Eg_mean_mm:.3f}")
    print(f"  max   = {Eg_max_mm:.3f}")
    print("--------------------------------------")
    print("E_mpbpe (Root-relative, 单位 mm):")
    print(f"  min   = {E_min_mm:.3f}")
    print(f"  median= {E_med_mm:.3f}")
    print(f"  mean  = {E_mean_mm:.3f}")
    print(f"  max   = {E_max_mm:.3f}")
    print("======================================")

if __name__ == "__main__":
    main()


"""
python tools/eval_mpbpe.py \
    --pkl retargeting_result/net_out_pkls_refine/aiming2_subject5_refine.pkl \
    --robot_xml assets/unitree_g1/g1_mocap_29dof.xml \
    --ik_config general_motion_retargeting/ik_configs/bvh_to_g1.json \
    --tables both

    """