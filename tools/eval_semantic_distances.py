#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
计算 retargeting 的“语义保持矩阵”：

- D_human (N x N): 人体各评估关节之间的平均距离
- D_robot (N x N): 机器人各评估关节之间的平均距离
- ratio  (N x N): D_robot / D_human
- D_diff (N x N): D_robot - D_human

距离单位：米；ratio 为无量纲；D_diff > 0 表示机器人对应关节间距更大。

用法示例：

    python tools/eval_semantic_distances.py \
        --pkl retargeting_result/lafan1_g1_save/aiming1_subject4_c.pkl \
        --robot_xml assets/unitree_g1/g1_mocap_29dof.xml \
        --ik_config general_motion_retargeting/ik_configs/bvh_to_g1.json \
        --tables both \
        --save_prefix semantic_mats/aiming1_subject4

会在终端打印四个带关节名的矩阵，并可选保存为：
    *_D_human.npy
    *_D_robot.npy
    *_ratio.npy
    *_D_diff.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco as mj


# ========== 基础工具函数 ==========

def load_robot_and_human_from_pkl(pkl_path):
    """从 pkl 中读取 robot_motion 和 human_motion。"""
    import pickle
    pkl_path = Path(pkl_path)
    with pkl_path.open("rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError("pkl 顶层不是 dict，无法解析。")

    if "robot_motion" not in data or "human_motion" not in data:
        raise ValueError("pkl 顶层缺少 robot_motion 或 human_motion。")

    return data["robot_motion"], data["human_motion"]


def build_robot_qpos(robot_motion):
    """
    由 robot_motion 还原 qpos 序列 (T, nq)：
    - root_pos: (T,3)
    - root_rot: (T,4) 存的是 xyzw
    - dof_pos : (T, ndof)
    -> qpos = [ root_pos , root_quat(wxyz) , dof_pos ]
    """
    root_pos = np.asarray(robot_motion["root_pos"], dtype=float)       # (T,3)
    root_rot_xyzw = np.asarray(robot_motion["root_rot"], dtype=float) # (T,4)
    dof_pos = np.asarray(robot_motion["dof_pos"], dtype=float)        # (T,ndof)

    T = root_pos.shape[0]
    assert root_rot_xyzw.shape[0] == T and dof_pos.shape[0] == T

    nq = 3 + 4 + dof_pos.shape[1]
    qpos = np.zeros((T, nq), dtype=float)

    qpos[:, :3] = root_pos
    # xyzw -> wxyz
    qpos[:, 3:7] = root_rot_xyzw[:, [3, 0, 1, 2]]
    qpos[:, 7:] = dof_pos

    return qpos


def fk_robot_bodies(model, qpos_seq, robot_body_names):
    """
    用 MuJoCo FK 求出给定 body 的世界坐标。

    model: mj.MjModel
    qpos_seq: (T, nq)
    robot_body_names: list[str]

    返回：
        xpos: (T, N, 3)
    """
    data = mj.MjData(model)
    T = qpos_seq.shape[0]
    N = len(robot_body_names)

    body_ids = []
    for name in robot_body_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in model.")
        body_ids.append(bid)

    xpos = np.zeros((T, N, 3), dtype=float)

    for t in range(T):
        data.qpos[:] = qpos_seq[t]
        mj.mj_forward(model, data)
        for j, bid in enumerate(body_ids):
            xpos[t, j] = data.xpos[bid]

    return xpos


def build_pairs_all_from_ik_config(ik_cfg, tables="both"):
    """
    从 bvh_to_g1.json 中构造 robot ↔ human 的“评估关节对”。

    注意：这里 **不按 pos_weight 过滤**，
    只要在 ik_match_table1/2 里出现过，就纳入评估，
    这样可以包含 torso_link <-> Spine2 等 pos_weight=0 的关节。

    返回：
        pairs: list[(robot_body_name, human_name)]
    """
    use_table1 = tables in ("1", "both") and ik_cfg.get("use_ik_match_table1", True)
    use_table2 = tables in ("2", "both") and ik_cfg.get("use_ik_match_table2", True)

    pairs = []
    seen = set()

    if use_table1 and "ik_match_table1" in ik_cfg:
        for robot_body_name, entry in ik_cfg["ik_match_table1"].items():
            human_name = entry[0]
            key = (robot_body_name, human_name)
            if key not in seen:
                pairs.append(key)
                seen.add(key)

    if use_table2 and "ik_match_table2" in ik_cfg:
        for robot_body_name, entry in ik_cfg["ik_match_table2"].items():
            human_name = entry[0]
            key = (robot_body_name, human_name)
            if key not in seen:
                pairs.append(key)
                seen.add(key)

    if not pairs:
        raise ValueError("在 ik_match_table 中没有找到任何关节对，无法构造评估关节对。")

    return pairs


# ========== 语义距离矩阵计算 ==========

def compute_distance_matrices(xpos_human, xpos_robot):
    """
    根据人和机器人的关节位置轨迹，计算四类 N x N 矩阵：

    - D_human[i,j] = mean_t || p_human(t,i) - p_human(t,j) ||
    - D_robot[i,j] = mean_t || p_robot(t,i) - p_robot(t,j) ||
    - ratio[i,j]   = D_robot[i,j] / D_human[i,j] (加一个很小的 eps 防 0)
    - D_diff[i,j]  = D_robot[i,j] - D_human[i,j]

    参数：
        xpos_human: (T, N, 3)
        xpos_robot: (T, N, 3)

    返回：
        D_human, D_robot, ratio, D_diff   (全是 (N,N) 的 numpy 数组)
    """
    T, N, _ = xpos_human.shape
    assert xpos_robot.shape == xpos_human.shape

    # 扩展维度后做广播，得到 (T, N, N, 3)
    H_i = xpos_human[:, :, None, :]   # (T, N, 1, 3)
    H_j = xpos_human[:, None, :, :]   # (T, 1, N, 3)
    diff_h = H_i - H_j                # (T, N, N, 3)

    R_i = xpos_robot[:, :, None, :]
    R_j = xpos_robot[:, None, :, :]
    diff_r = R_i - R_j

    # 每一帧的两两距离 (T, N, N)
    dist_h_t = np.linalg.norm(diff_h, axis=-1)
    dist_r_t = np.linalg.norm(diff_r, axis=-1)

    # 对时间求平均，得到 (N, N)
    D_human = dist_h_t.mean(axis=0)
    D_robot = dist_r_t.mean(axis=0)

    # 比值矩阵
    eps = 1e-8
    ratio = D_robot / (D_human + eps)

    # 差值矩阵
    D_diff = D_robot - D_human

    # 对角线上 D_human = D_robot = 0，ratio 理论上无意义，这里设成 1 方便查看
    np.fill_diagonal(ratio, 1.0)

    return D_human, D_robot, ratio, D_diff

def _vectorize_offdiag(mat, use_upper=True):
    """取距离矩阵的非对角元素向量（默认取上三角，避免重复计数）。"""
    N = mat.shape[0]
    if use_upper:
        ii, jj = np.triu_indices(N, k=1)
    else:
        ii, jj = np.where(~np.eye(N, dtype=bool))
    return mat[ii, jj].astype(float)


def _rankdata_average_ties(x):
    """
    简易 rankdata（处理 ties 用平均秩），用于 Spearman。
    返回从 1 开始的 rank。
    """
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]

    ranks_sorted = np.empty_like(x_sorted, dtype=float)

    n = len(x_sorted)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x_sorted[j + 1] == x_sorted[i]:
            j += 1
        # i..j 是一组相等值，用平均秩（注意秩从 1 开始）
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        ranks_sorted[i:j + 1] = avg_rank
        i = j + 1

    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def _pearson_corr(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def _cosine_sim(a, b, eps=1e-12):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    return float(np.dot(a, b) / denom)


def compute_matrix_similarity(D_human, D_robot):
    """
    计算 D_human 与 D_robot 的整体相似度（基于非对角元素向量化）。
    返回 dict，含相关系数、误差、最佳尺度拟合等。
    """
    vh = _vectorize_offdiag(D_human, use_upper=True)
    vr = _vectorize_offdiag(D_robot, use_upper=True)

    eps = 1e-12

    pearson = _pearson_corr(vh, vr, eps=eps)

    rh = _rankdata_average_ties(vh)
    rr = _rankdata_average_ties(vr)
    spearman = _pearson_corr(rh, rr, eps=eps)

    cosine = _cosine_sim(vh, vr, eps=eps)

    diff = vr - vh
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    rel_fro = float(np.linalg.norm(diff) / (np.linalg.norm(vh) + eps))

    # 最佳尺度拟合：min_alpha || alpha * vh - vr ||_2
    alpha = float(np.dot(vh, vr) / (np.dot(vh, vh) + eps))
    scale_inv_rel = float(np.linalg.norm(alpha * vh - vr) / (np.linalg.norm(vr) + eps))

    return {
        "pearson": pearson,                 # 越接近 1 越好
        "spearman": spearman,               # 越接近 1 越好
        "cosine": cosine,                   # 越接近 1 越好
        "mae_m": mae,                       # 单位 m，越小越好
        "rmse_m": rmse,                     # 单位 m，越小越好
        "rel_fro": rel_fro,                 # 无量纲，越小越好
        "alpha_best_scale": alpha,          # 尺度比（人->机）
        "scale_invariant_rel_error": scale_inv_rel,  # 无量纲，越小越好
        "num_pairs": int(len(vh)),          # 用了多少个(i,j)对
    }




def print_labeled_matrix(mat, joint_names, title, float_fmt="{:.4f}"):
    """
    打印一个带行列关节名的矩阵：
    第一行打印列关节名，第一列打印行关节名。
    """
    N = len(joint_names)
    assert mat.shape == (N, N)

    print(f"\n=== {title} (N={N}) ===")
    # 打印列名
    header = ["{:>15}".format("")]  # 左上角空白
    for name in joint_names:
        header.append("{:>15}".format(name))
    print("".join(header))

    # 每一行：行名 + 每个元素
    for i in range(N):
        row = ["{:>15}".format(joint_names[i])]
        for j in range(N):
            row.append("{:>15}".format(float_fmt.format(mat[i, j])))
        print("".join(row))


# ========== main ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=str, required=True,
                        help="包含 robot_motion 和 human_motion 的 pkl 路径")
    parser.add_argument("--robot_xml", type=str, required=True,
                        help="MuJoCo 机器人 xml 路径（例如 g1_mocap_29dof.xml）")
    parser.add_argument("--ik_config", type=str, required=True,
                        help="IK 配置 json（例如 bvh_to_g1.json）")
    parser.add_argument("--tables", type=str, default="both",
                        choices=["1", "2", "both"],
                        help="使用 ik_match_table1 / 2 / both 中的哪些关节（默认 both）")
    parser.add_argument("--save_prefix", type=str, default=None,
                        help="若指定，则将 D_human, D_robot, ratio, D_diff 以 .npy 格式保存，"
                             "文件名为 '<save_prefix>_D_human.npy' 等。")
    args = parser.parse_args()

    # 1) 读取 pkl
    robot_motion, human_motion = load_robot_and_human_from_pkl(args.pkl)
    if human_motion is None:
        raise ValueError("pkl 中 human_motion 为 None，无法进行语义距离计算。")

    # 2) MuJoCo 模型 & 机器人 qpos
    model = mj.MjModel.from_xml_path(args.robot_xml)
    qpos_robot = build_robot_qpos(robot_motion)   # (T_r, nq)

    # 3) 人体 anchor 世界坐标
    root_pos_h = np.asarray(human_motion["root_pos"], dtype=float)          # (T_h,3)
    local_body_pos = np.asarray(human_motion["local_body_pos"], dtype=float)# (T_h,Nh,3)
    human_body_names = list(human_motion["link_body_list"])                 # 长度 Nh

    T_h, Nh, _ = local_body_pos.shape
    human_global_pos = local_body_pos + root_pos_h[:, None, :]  # (T_h,Nh,3)

    # 4) 读 ik_config，构造“所有”（不按 pos_weight 过滤）的关节对
    with open(args.ik_config, "r", encoding="utf-8") as f:
        ik_cfg = json.load(f)

    pairs = build_pairs_all_from_ik_config(ik_cfg, tables=args.tables)

    robot_body_names = []
    human_indices = []
    for robot_body_name, human_name in pairs:
        if human_name not in human_body_names:
            raise ValueError(f"Human 关节 {human_name} 不在 human_motion['link_body_list'] 中。")
        idx_h = human_body_names.index(human_name)
        robot_body_names.append(robot_body_name)
        human_indices.append(idx_h)

    N = len(robot_body_names)

    print("评估关节对（robot ↔ human），数量 N =", N)
    for j in range(N):
        print(f"  {j:02d}: {robot_body_names[j]}  <->  {human_body_names[human_indices[j]]}")

    # 5) 对齐时间长度
    T_r = qpos_robot.shape[0]
    T = min(T_r, T_h)
    if T < T_r or T < T_h:
        print(f"[Info] robot T={T_r}, human T={T_h}，对齐到 T={T}")
    qpos_robot = qpos_robot[:T]
    human_global_pos = human_global_pos[:T]

    # 6) FK 机器人 body 的世界坐标 (T,N,3)
    xpos_robot = fk_robot_bodies(model, qpos_robot, robot_body_names)

    # 7) 人体对应 anchor 的世界坐标 (T,N,3)
    xpos_human = np.zeros((T, N, 3), dtype=float)
    for j, idx_h in enumerate(human_indices):
        xpos_human[:, j, :] = human_global_pos[:, idx_h, :]

    # 8) 计算四个 N x N 矩阵
    D_human, D_robot, ratio, D_diff = compute_distance_matrices(xpos_human, xpos_robot)


    # ✅ 只用参与评估的 N 个关节名做标签（避免 Nh != N）
    joint_names = [human_body_names[i] for i in human_indices]

    # ✅ 新增：整体相似度指标（基于非对角上三角）
    sim = compute_matrix_similarity(D_human, D_robot)
    print("\n=== D_human vs D_robot 相似度（基于上三角非对角元素） ===")
    for k, v in sim.items():
        if isinstance(v, float):
            print(f"  {k:>26s}: {v:.6f}")
        else:
            print(f"  {k:>26s}: {v}")

    print("\n=== 语义距离矩阵形状 ===")
    print("D_human shape:", D_human.shape)
    print("D_robot shape:", D_robot.shape)
    print("ratio   shape:", ratio.shape)
    print("D_diff  shape:", D_diff.shape)

    joint_names = human_body_names  # 行列标签用 human 这边的名字即可

    # 9) 打印带关节名的四个矩阵
    print_labeled_matrix(D_human, joint_names, "D_human（人体关节平均距离，单位：m）")
    print_labeled_matrix(D_robot, joint_names, "D_robot（机器人关节平均距离，单位：m）")
    print_labeled_matrix(ratio, joint_names, "ratio = D_robot / D_human（无量纲）")
    print_labeled_matrix(D_diff, joint_names, "D_diff = D_robot - D_human（距离差，单位：m）")

    # 可选保存
    if args.save_prefix is not None:
        save_prefix = Path(args.save_prefix)
        save_prefix.parent.mkdir(parents=True, exist_ok=True)

        np.save(save_prefix.with_name(save_prefix.name + "_D_human.npy"), D_human)
        np.save(save_prefix.with_name(save_prefix.name + "_D_robot.npy"), D_robot)
        np.save(save_prefix.with_name(save_prefix.name + "_ratio.npy"),    ratio)
        np.save(save_prefix.with_name(save_prefix.name + "_D_diff.npy"),   D_diff)

        print(f"\n矩阵已保存到：")
        print(f"  {save_prefix.name}_D_human.npy")
        print(f"  {save_prefix.name}_D_robot.npy")
        print(f"  {save_prefix.name}_ratio.npy")
        print(f"  {save_prefix.name}_D_diff.npy")
    import json as _json
    with open(save_prefix.with_name(save_prefix.name + "_similarity.json"), "w", encoding="utf-8") as f:
        _json.dump(sim, f, ensure_ascii=False, indent=2)
        
    # 简单看一下 ratio / D_diff 的统计
    print("\n=== ratio 统计（D_robot / D_human） ===")
    ratio_flat = ratio.reshape(-1)
    print(f"  min   = {ratio_flat.min():.4f}")
    print(f"  max   = {ratio_flat.max():.4f}")
    print(f"  mean  = {ratio_flat.mean():.4f}")

    print("\n=== D_diff 统计（D_robot - D_human，单位：m） ===")
    diff_flat = D_diff.reshape(-1)
    print(f"  min   = {diff_flat.min():.4f}")
    print(f"  max   = {diff_flat.max():.4f}")
    print(f"  mean  = {diff_flat.mean():.4f}")
    print("=========================================")


if __name__ == "__main__":
    main()
