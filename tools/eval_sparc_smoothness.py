#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对 GMR 保存的 pkl 文件，计算每个“评估关节对”（robot body ↔ human anchor）的平滑度（SPARC）。

用法示例：

    python tools/eval_sparc_smoothness.py \
        --pkl retargeting_result/lafan1_g1/aiming1_subject1_refined.pkl \
        --robot_xml assets/unitree_g1/g1_mocap_29dof.xml \
        --ik_config general_motion_retargeting/ik_configs/bvh_to_g1.json \
        --tables both

SPARC 参考论文：
S. Balasubramanian et al., "On the analysis of movement smoothness",
Journal of NeuroEngineering and Rehabilitation, 2015.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import mujoco as mj


# ===================== SPARC 实现 =====================

def _sparc_1d_speed(speed, fs, padlevel=4, fc_max=10.0, amp_th=0.05):
    """
    对 1D 速度标量序列计算 SPARC。

    speed: (T,) 速度标量（非负），采样频率 fs (Hz)
    padlevel: 频谱插值倍数（通过零填充提高频率分辨率）
    fc_max: 最高考虑频率（Hz），一般 10~20
    amp_th: 归一化谱幅的阈值 V

    返回：
        sparc: float，数值越接近 0（注意是负数）表示越平滑。
    """
    speed = np.asarray(speed, dtype=float)

    # 全零或常数轨迹没有实际意义
    if np.allclose(speed, speed[0]):
        return np.nan

    T = len(speed)
    # 零填充，提升频率分辨率
    n_fft = int(2 ** np.ceil(np.log2(T * padlevel)))

    # FFT -> 幅度谱
    v_fft = np.fft.rfft(speed, n=n_fft)          # 只要非负频率
    mag = np.abs(v_fft)                          # (Nf,)

    # DC 分量为 0 时无法归一化
    if mag[0] <= 0:
        return np.nan

    # 归一化幅度谱（除以 DC 分量）
    vhat = mag / mag[0]

    # 频率轴（Hz -> rad/s）
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)   # (Nf,)
    omegas = 2.0 * np.pi * freqs                 # (Nf,)

    # 限制最高频率索引
    idx_fc_max = np.searchsorted(freqs, fc_max, side="right") - 1
    idx_fc_max = max(1, idx_fc_max)

    # 找到最后一个幅度还 >= 阈值的位置
    above_th = np.where(vhat >= amp_th)[0]
    if len(above_th) == 0:
        idx_thr = 1
    else:
        idx_thr = int(above_th[-1])

    idx_c = int(min(idx_fc_max, idx_thr))
    idx_c = max(1, idx_c)  # 至少有两个点

    # 截到 [0, ω_c]，再对频率归一化到 [0,1]
    omega_c = omegas[idx_c]
    x = omegas[:idx_c + 1] / omega_c  # 0~1
    y = vhat[:idx_c + 1]

    dx = np.diff(x)
    dy = np.diff(y)

    L = np.sum(np.sqrt(dx ** 2 + dy ** 2))
    sparc = -float(L)  # 弧长越小越平滑，加负号后越接近 0 越平滑

    return sparc


def sparc_from_3d_position(pos, fs, padlevel=4, fc_max=10.0, amp_th=0.05):
    """
    对 3D 位置轨迹 pos(t) 计算 SPARC：
        1. 先求 3D 速度向量 v(t) = d pos / dt
        2. 再取速度标量 ||v(t)||，丢给 _sparc_1d_speed

    pos: (T,3)
    fs:  采样频率 Hz
    """
    pos = np.asarray(pos, dtype=float)
    assert pos.ndim == 2 and pos.shape[1] == 3, "pos 必须是 (T,3)"

    dt = 1.0 / fs
    vel = np.gradient(pos, dt, axis=0)          # (T,3)
    speed = np.linalg.norm(vel, axis=1)        # (T,)

    return _sparc_1d_speed(speed, fs, padlevel=padlevel, fc_max=fc_max, amp_th=amp_th)


# ===================== GMR 相关工具函数 =====================

def load_robot_and_human_from_pkl(pkl_path):
    """从 pkl 中读取 robot_motion 和 human_motion。"""
    pkl_path = Path(pkl_path)
    import pickle
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
    root_pos = np.asarray(robot_motion["root_pos"], dtype=float)      # (T,3)
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
        xpos: (T, J, 3)
    """
    data = mj.MjData(model)

    T = qpos_seq.shape[0]
    print("-----------------------------")
    print(T)
    print(qpos_seq.shape)
    print(robot_body_names)
    J = len(robot_body_names)

    body_ids = []
    for name in robot_body_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in model.")
        body_ids.append(bid)

    xpos = np.zeros((T, J, 3), dtype=float)

    for t in range(T):
        data.qpos[:] = qpos_seq[t]
        mj.mj_forward(model, data)
        for j, bid in enumerate(body_ids):
            xpos[t, j] = data.xpos[bid]

    return xpos


def build_pairs_from_ik_config(ik_cfg, tables="both"):
    """
    从 bvh_to_g1.json 中构造 robot ↔ human 的“评估关节对”。

    返回：
        pairs: list[(robot_body_name, human_name, pos_weight)]
    """
    use_table1 = tables in ("1", "both") and ik_cfg.get("use_ik_match_table1", True)
    use_table2 = tables in ("2", "both") and ik_cfg.get("use_ik_match_table2", True)

    pairs = []
    seen = set()

    if use_table1:
        for robot_body_name, entry in ik_cfg["ik_match_table1"].items():
            human_name, pos_w, rot_w, pos_off, rot_off = entry
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
        raise ValueError("在 ik_match_table 中没有找到 pos_weight > 0 的关节，无法构造评估关节对。")

    return pairs


# ===================== main：计算 SPARC =====================

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
    parser.add_argument("--fc_max", type=float, default=10.0,
                        help="SPARC 计算时的最高考虑频率 fc_max (Hz)，默认 10.0")
    parser.add_argument("--amp_th", type=float, default=0.05,
                        help="SPARC 谱幅阈值 amp_th，默认 0.05")
    args = parser.parse_args()
    print("-----------------------")

    # 1) 读取 pkl
    robot_motion, human_motion = load_robot_and_human_from_pkl(args.pkl)
    if human_motion is None:
        raise ValueError("pkl 中 human_motion 为 None，无法计算 human 的平滑度。")

    fps_h = human_motion.get("fps", None)
    fps_r = robot_motion.get("fps", None)
    if fps_h is None and fps_r is None:
        raise ValueError("找不到 fps，请确认 pkl 保存时包含 'fps' 字段。")
    fps = fps_h if fps_h is not None else fps_r

    print(f"[Info] 使用 fps = {fps} Hz")

    # 2) MuJoCo 模型 & 机器人 qpos
    model = mj.MjModel.from_xml_path(args.robot_xml)
    qpos_robot = build_robot_qpos(robot_motion)   # (T_r, nq)

    # 3) 人体 anchor 世界坐标
    root_pos_h = np.asarray(human_motion["root_pos"], dtype=float)           # (T_h,3)
    local_body_pos = np.asarray(human_motion["local_body_pos"], dtype=float) # (T_h,Nh,3)
    human_body_names = list(human_motion["link_body_list"])                  # 长度 Nh

    T_h, Nh, _ = local_body_pos.shape
    human_global_pos = local_body_pos + root_pos_h[:, None, :]  # (T_h,Nh,3)

    # 4) 读取 ik_config，构造评估关节对
    with open(args.ik_config, "r", encoding="utf-8") as f:
        ik_cfg = json.load(f)

    pairs = build_pairs_from_ik_config(ik_cfg, tables=args.tables)

    robot_body_names = []
    human_indices = []
    joint_pairs = []

    for robot_body_name, human_name, pos_w in pairs:
        if human_name not in human_body_names:
            raise ValueError(f"Human 关节 {human_name} 不在 human_motion['link_body_list'] 中。")
        idx_h = human_body_names.index(human_name)

        robot_body_names.append(robot_body_name)
        human_indices.append(idx_h)
        joint_pairs.append((robot_body_name, human_name))

    J = len(robot_body_names)

    print("评估关节对（robot ↔ human）：")
    for j in range(J):
        print(f"  {j:02d}: {robot_body_names[j]}  <->  {human_body_names[human_indices[j]]}")

    # 5) 对齐时间长度
    T_r = qpos_robot.shape[0]
    T = min(T_r, T_h)
    if T < T_r or T < T_h:
        print(f"[Info] robot T={T_r}, human T={T_h}，对齐到 T={T}")

    qpos_robot = qpos_robot[:T]
    human_global_pos = human_global_pos[:T]

    # 6) FK 机器人 body 的世界坐标
    xpos_robot_all = fk_robot_bodies(model, qpos_robot, robot_body_names)  # (T,J,3)

    # 7) 人体对应 anchor 的世界坐标
    xpos_human = np.zeros((T, J, 3), dtype=float)
    for j, idx_h in enumerate(human_indices):
        xpos_human[:, j, :] = human_global_pos[:, idx_h, :]

    # 8) 每个关节对分别计算 SPARC
    human_sparc_per_joint = []
    robot_sparc_per_joint = []

    print("\n=== 每个评估关节对的 SPARC 平滑度（越接近 0 越平滑，SPARC 通常为负数） ===")
    for j in range(J):
        name_robot, name_human = joint_pairs[j]

        pos_h = xpos_human[:, j, :]       # (T,3)
        pos_r = xpos_robot_all[:, j, :]   # (T,3)

        sparc_h = sparc_from_3d_position(
            pos_h, fps, padlevel=4, fc_max=args.fc_max, amp_th=args.amp_th
        )
        sparc_r = sparc_from_3d_position(
            pos_r, fps, padlevel=4, fc_max=args.fc_max, amp_th=args.amp_th
        )

        human_sparc_per_joint.append(sparc_h)
        robot_sparc_per_joint.append(sparc_r)

        print(f"[{j:02d}] {name_robot}  <->  {name_human}")
        print(f"     human SPARC = {sparc_h:.4f}")
        print(f"     robot SPARC = {sparc_r:.4f}")

    human_sparc_per_joint = np.array(human_sparc_per_joint, dtype=float)
    robot_sparc_per_joint = np.array(robot_sparc_per_joint, dtype=float)

    # 9) 汇总统计
    def summarize(arr, label):
        arr_valid = arr[~np.isnan(arr)]
        if arr_valid.size == 0:
            print(f"{label}: 全部为 NaN，无法统计。")
            return
        print(f"{label}:")
        print(f"  min    = {float(arr_valid.min()):.4f}")
        print(f"  median = {float(np.median(arr_valid)):.4f}")
        print(f"  mean   = {float(arr_valid.mean()):.4f}")
        print(f"  max    = {float(arr_valid.max()):.4f}")

    print("\n=== 汇总统计（所有评估关节 SPARC） ===")
    summarize(human_sparc_per_joint, "Human joints SPARC")
    summarize(robot_sparc_per_joint, "Robot joints SPARC")
    print("=================================================")


if __name__ == "__main__":
    main()
