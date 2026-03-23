#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
refine_collision_minchange_hinge.py

对输入 pkl（原始 GMR 输出或已 refine 输出都可）进行“碰撞安全裕量 hinge”精修：
  - C: 使用 hinge 安全裕量：sum max(0, m - dist)^2，推动 dist >= m（默认 m=0.002m）
  - 只修复 dist < m 的帧（否则 0.001m 的“近接触”不会被处理）
  - 每个碰撞帧只在“碰撞相关链条 DOFs”上做最小改动搜索（0, +delta, -delta）
  - 手部碰撞优先（若接触涉及 hand/wrist body，则只用该手部链条的 DOFs）
  - 为保证连续性：默认把一次角度更新扩散到 t-1/t/t+1（权重 0.25/0.5/0.25）

用法（直接对原始 GMR pkl）：
  python tools/refine_collision_minchange_hinge.py \
    --input_pkl  retargeting_result/lafan1_g1_save/aiming1_subject4_c.pkl \
    --robot_xml  assets/unitree_g1/g1_mocap_29dof.xml \
    --ik_config  general_motion_retargeting/ik_configs/bvh_to_g1.json \
    --tables     both \
    --output_pkl retargeting_result/lafan1_g1_save/aiming1_subject4_refine.pkl 

建议（你的场景：只剩少量 1mm 级别碰撞）：
  coll_margin=0.002（2mm） 或 0.0015（1.5mm）
  delta_angle=0.01~0.02
  w_M/w_dev 适当调大，避免抖动
"""

import argparse
import json
from pathlib import Path
import pickle
import time
from typing import List, Tuple, Dict

import numpy as np
import mujoco as mj


# ------------------------- qpos / motion -------------------------

def build_robot_qpos_template(robot_motion: dict) -> np.ndarray:
    """
    由 robot_motion 构造 qpos_template (T,nq)：只固定 root (pos+quat)，dof 将在运行时覆盖。
    robot_motion:
      - root_pos: (T,3)
      - root_rot: (T,4) xyzw
      - dof_pos : (T,ndof)
    """
    root_pos = np.asarray(robot_motion["root_pos"], dtype=float)
    root_rot_xyzw = np.asarray(robot_motion["root_rot"], dtype=float)
    dof_pos = np.asarray(robot_motion["dof_pos"], dtype=float)

    T = root_pos.shape[0]
    assert root_rot_xyzw.shape[0] == T and dof_pos.shape[0] == T

    nq = 7 + dof_pos.shape[1]
    qpos_template = np.zeros((T, nq), dtype=float)
    qpos_template[:, :3] = root_pos
    # xyzw -> wxyz (MuJoCo)
    qpos_template[:, 3:7] = root_rot_xyzw[:, [3, 0, 1, 2]]
    # 7: 后续填 dof_pos
    return qpos_template


# ------------------------- IK config -> anchor pairs -------------------------

def build_pairs_from_ik_config(ik_cfg: dict, tables: str = "both") -> List[Tuple[str, str, float]]:
    """
    从 bvh_to_g1.json 中构造 robot ↔ human 的“评估关节对”。
    返回：[(robot_body_name, human_name, pos_weight), ...]
    """
    use_table1 = tables in ("1", "both") and ik_cfg.get("use_ik_match_table1", True)
    use_table2 = tables in ("2", "both") and ik_cfg.get("use_ik_match_table2", True)

    pairs = []
    seen = set()

    if use_table1 and "ik_match_table1" in ik_cfg:
        for robot_body_name, entry in ik_cfg["ik_match_table1"].items():
            human_name, pos_w, rot_w, pos_off, rot_off = entry
            if pos_w >= 0:
                key = (robot_body_name, human_name)
                if key not in seen:
                    pairs.append((robot_body_name, human_name, pos_w))
                    seen.add(key)

    if use_table2 and "ik_match_table2" in ik_cfg:
        for robot_body_name, entry in ik_cfg["ik_match_table2"].items():
            human_name, pos_w, rot_w, pos_off, rot_off = entry
            if pos_w > 0:
                key = (robot_body_name, human_name)
                if key not in seen:
                    pairs.append((robot_body_name, human_name, pos_w))
                    seen.add(key)

    if not pairs:
        raise ValueError("在 ik_match_table 中没有找到 pos_weight > 0 的关节对。")
    return pairs


# ------------------------- body -> DOFs (ordered) -------------------------

def build_body_influence_dofs_ordered(model: mj.MjModel, qpos_root_size: int = 7) -> List[List[int]]:
    """
    对于每个 body，返回一个“能影响它的 dof 索引列表”，按“从该 body 向上到 root”的顺序排列（更偏局部/末端优先）。
    这会自然包含：手 -> 前臂 -> 上臂 -> 躯干 -> 骨盆 ... 的顺序。
    """
    nbody = model.nbody
    njnt = model.njnt
    nq = model.nq

    body_parent = model.body_parentid
    body_jntnum = model.body_jntnum
    body_jntadr = model.body_jntadr
    jnt_qposadr = model.jnt_qposadr

    # 每个 joint 占用的 qpos 维数（通过 qposadr 差分）
    num_qpos_per_jnt = np.zeros(njnt, dtype=int)
    for j in range(njnt):
        start = jnt_qposadr[j]
        end = jnt_qposadr[j + 1] if j + 1 < njnt else nq
        num_qpos_per_jnt[j] = end - start

    influence = [[] for _ in range(nbody)]

    for b in range(nbody):
        cur = b
        seen = set()
        while cur > 0:
            jnum = body_jntnum[cur]
            jadr = body_jntadr[cur]
            for k in range(jnum):
                j = jadr + k
                start = jnt_qposadr[j]
                end = start + num_qpos_per_jnt[j]
                for q_idx in range(start, end):
                    if q_idx >= qpos_root_size:
                        dof_idx = q_idx - qpos_root_size
                        if dof_idx not in seen:
                            influence[b].append(dof_idx)
                            seen.add(dof_idx)
            cur = body_parent[cur]

    return influence


# ------------------------- collision scan with hinge margin -------------------------

def collect_near_contact_frames(model: mj.MjModel,
                                qpos_template: np.ndarray,
                                dof_pos: np.ndarray,
                                coll_margin: float) -> List[List[Dict]]:
    """
    扫描整条轨迹，记录所有“dist < coll_margin”的机器人内部接触（排除 world）。
    返回 frame_contacts[t] = [ {body1, body2, body1_name, body2_name, dist}, ... ]
    """
    T = dof_pos.shape[0]
    data = mj.MjData(model)
    frame_contacts = [[] for _ in range(T)]

    for t in range(T):
        qpos = qpos_template[t].copy()
        qpos[7:] = dof_pos[t]
        data.qpos[:] = qpos
        mj.mj_forward(model, data)

        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if g1 < 0 or g2 < 0:
                continue

            b1 = model.geom_bodyid[g1]
            b2 = model.geom_bodyid[g2]
            n1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b1)
            n2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b2)

            if n1 == "world" or n2 == "world":
                continue

            dist = float(c.dist)
            if dist < coll_margin:
                frame_contacts[t].append({
                    "body1": int(b1),
                    "body2": int(b2),
                    "body1_name": n1,
                    "body2_name": n2,
                    "dist": dist,
                })

    return frame_contacts


def get_target_dofs_for_frame(t: int,
                              frame_contacts: List[List[Dict]],
                              influence_dofs: List[List[int]],
                              hand_keywords=("hand", "wrist")) -> List[int]:
    """
    对于帧 t：
      - 若碰撞涉及 hand/wrist body，则只用这些 body（但其 influence 会包含肩/肘等上游 DOFs）
      - 否则用所有碰撞涉及 body
    返回 target_dofs（有序，末端优先）
    """
    contacts = frame_contacts[t]
    if not contacts:
        return []

    hand_bodies = []
    other_bodies = []
    for c in contacts:
        for bid, name in [(c["body1"], c["body1_name"]), (c["body2"], c["body2_name"])]:
            lname = name.lower()
            if any(k in lname for k in hand_keywords):
                if bid not in hand_bodies:
                    hand_bodies.append(bid)
            else:
                if bid not in other_bodies:
                    other_bodies.append(bid)

    bodies = hand_bodies if hand_bodies else (hand_bodies + other_bodies)

    # 末端优先：按 body influence 的顺序拼起来去重
    out = []
    seen = set()
    for b in bodies:
        for d in influence_dofs[b]:
            if d not in seen:
                out.append(d)
                seen.add(d)
    return out


# ------------------------- metrics: collision hinge + semantics + continuity -------------------------

def collision_hinge_sum(model: mj.MjModel,
                        data: mj.MjData,
                        qpos_template: np.ndarray,
                        dof_pos: np.ndarray,
                        frames: List[int],
                        coll_margin: float) -> float:
    """
    对 frames 中每帧，计算 hinge 碰撞代价：sum max(0, m - dist)^2
    """
    m = float(coll_margin)
    acc = 0.0
    for t in frames:
        qpos = qpos_template[t].copy()
        qpos[7:] = dof_pos[t]
        data.qpos[:] = qpos
        mj.mj_forward(model, data)

        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if g1 < 0 or g2 < 0:
                continue

            b1 = model.geom_bodyid[g1]
            b2 = model.geom_bodyid[g2]
            n1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b1)
            n2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b2)
            if n1 == "world" or n2 == "world":
                continue

            dist = float(c.dist)
            pen = max(0.0, m - dist)
            acc += pen * pen
    return acc


def get_robot_anchor_positions(model: mj.MjModel,
                              data: mj.MjData,
                              qpos_template: np.ndarray,
                              dof_pos: np.ndarray,
                              t: int,
                              body_ids: List[int]) -> np.ndarray:
    """
    返回机器人在帧 t 的 anchor body 世界坐标：(J,3)
    """
    qpos = qpos_template[t].copy()
    qpos[7:] = dof_pos[t]
    data.qpos[:] = qpos
    mj.mj_forward(model, data)

    J = len(body_ids)
    x = np.zeros((J, 3), dtype=float)
    for j, bid in enumerate(body_ids):
        x[j] = data.xpos[bid]
    return x


def frame_semantic_direction_error(x_h: np.ndarray, x_r: np.ndarray, root_index: int) -> float:
    """
    S: 方向语义误差 = mean(1 - cosθ) over anchors
    x_h, x_r: (J,3)
    """
    eps = 1e-8
    vh = x_h - x_h[root_index]
    vr = x_r - x_r[root_index]

    nh = vh / (np.linalg.norm(vh, axis=-1, keepdims=True) + eps)
    nr = vr / (np.linalg.norm(vr, axis=-1, keepdims=True) + eps)

    dot = np.clip(np.sum(nh * nr, axis=-1), -1.0, 1.0)
    return float(np.mean(1.0 - dot))


def frame_root_relative_mpjpe(x_h: np.ndarray, x_r: np.ndarray, root_index: int) -> float:
    """
    D: root-relative MPJPE（一帧）= mean || (xr-rootr) - (xh-rooth) ||
    """
    vh = x_h - x_h[root_index]
    vr = x_r - x_r[root_index]
    diff = vr - vh
    return float(np.mean(np.linalg.norm(diff, axis=-1)))


def continuity_cost(dof_pos: np.ndarray, t: int, smooth_dofs: List[int]) -> float:
    """
    M: 对 smooth_dofs 计算一阶差分连续性（与前后帧的 MSE 平均）
    """
    T = dof_pos.shape[0]
    if not smooth_dofs:
        return 0.0

    acc = 0.0
    cnt = 0
    if t > 0:
        diff = dof_pos[t, smooth_dofs] - dof_pos[t - 1, smooth_dofs]
        acc += float(np.mean(diff ** 2))
        cnt += 1
    if t < T - 1:
        diff = dof_pos[t + 1, smooth_dofs] - dof_pos[t, smooth_dofs]
        acc += float(np.mean(diff ** 2))
        cnt += 1
    return acc / cnt if cnt > 0 else 0.0


def dev_cost(dof_pos: np.ndarray, dof_orig: np.ndarray, t: int, smooth_dofs: List[int]) -> float:
    """
    dev: 偏离原始 GMR 的 MSE（只在 smooth_dofs 上统计）
    """
    if not smooth_dofs:
        return 0.0
    diff = dof_pos[t, smooth_dofs] - dof_orig[t, smooth_dofs]
    return float(np.mean(diff ** 2))


# ------------------------- refine: min-change with hinge + temporal spread -------------------------

def apply_temporal_spread(dof_pos: np.ndarray,
                          t: int,
                          dof_idx: int,
                          delta: float,
                          enable: bool) -> List[Tuple[int, float]]:
    """
    将一次更新扩散到 t-1/t/t+1，以提升连续性：
      - t-1: 0.25*delta
      - t  : 0.50*delta
      - t+1: 0.25*delta
    返回实际修改的 (frame, applied_delta) 列表，方便撤销。
    """
    T = dof_pos.shape[0]
    if not enable:
        dof_pos[t, dof_idx] += delta
        return [(t, delta)]

    weights = [(-1, 0.25), (0, 0.50), (+1, 0.25)]
    applied = []
    for off, w in weights:
        tt = t + off
        if 0 <= tt < T:
            dd = w * delta
            dof_pos[tt, dof_idx] += dd
            applied.append((tt, dd))
    return applied


def undo_temporal_spread(dof_pos: np.ndarray, applied: List[Tuple[int, float]], dof_idx: int):
    for tt, dd in applied:
        dof_pos[tt, dof_idx] -= dd


def refine_frame(model: mj.MjModel,
                 data: mj.MjData,
                 qpos_template: np.ndarray,
                 dof_pos: np.ndarray,
                 dof_orig: np.ndarray,
                 human_anchor_pos: np.ndarray,
                 body_ids: List[int],
                 root_index: int,
                 frame_contacts: List[List[Dict]],
                 influence_dofs: List[List[int]],
                 t: int,
                 coll_margin: float,
                 delta_angle: float,
                 temporal_spread: bool,
                 w_C: float,
                 w_S: float,
                 w_D: float,
                 w_M: float,
                 w_dev: float,
                 improve_tol: float = 1e-12) -> None:
    """
    对单帧 t 做最小改动修复：
      - 目标：优先让 hinge 碰撞 C 下降（推动 dist >= margin）
      - 其次：在 C 差不多时，选择 S/D/M/dev 综合最小
    """
    if not frame_contacts[t]:
        return

    target_dofs = get_target_dofs_for_frame(t, frame_contacts, influence_dofs)
    if not target_dofs:
        return

    # 评估窗口：若启用 temporal_spread，则碰撞代价对 [t-1,t,t+1] 一起算，避免推给邻居
    frames_eval = [t]
    if temporal_spread:
        if t - 1 >= 0:
            frames_eval.append(t - 1)
        if t + 1 < dof_pos.shape[0]:
            frames_eval.append(t + 1)
        frames_eval = sorted(list(set(frames_eval)))

    # base metrics
    C_base = collision_hinge_sum(model, data, qpos_template, dof_pos, frames_eval, coll_margin)

    x_r = get_robot_anchor_positions(model, data, qpos_template, dof_pos, t, body_ids)
    x_h = human_anchor_pos[t]
    S_base = frame_semantic_direction_error(x_h, x_r, root_index)
    D_base = frame_root_relative_mpjpe(x_h, x_r, root_index)
    M_base = continuity_cost(dof_pos, t, target_dofs)
    dev_base = dev_cost(dof_pos, dof_orig, t, target_dofs)
    E_base = w_C * C_base + w_S * S_base + w_D * D_base + w_M * M_base + w_dev * dev_base

    # 对每个 DOF 贪心：尝试 0/+δ/-δ
    for d in target_dofs:
        best = (C_base, E_base, 0.0)  # 先按 C 最小，再按 E 最小
        # 试探三种候选
        for delta in (0.0, +delta_angle, -delta_angle):
            if delta == 0.0:
                C_now, E_now = C_base, E_base
            else:
                applied = apply_temporal_spread(dof_pos, t, d, delta, temporal_spread)

                C_now = collision_hinge_sum(model, data, qpos_template, dof_pos, frames_eval, coll_margin)

                # S/D 用中心帧 t
                x_r2 = get_robot_anchor_positions(model, data, qpos_template, dof_pos, t, body_ids)
                S_now = frame_semantic_direction_error(x_h, x_r2, root_index)
                D_now = frame_root_relative_mpjpe(x_h, x_r2, root_index)
                M_now = continuity_cost(dof_pos, t, target_dofs)
                dev_now = dev_cost(dof_pos, dof_orig, t, target_dofs)
                E_now = w_C * C_now + w_S * S_now + w_D * D_now + w_M * M_now + w_dev * dev_now

                undo_temporal_spread(dof_pos, applied, d)

            cand = (C_now, E_now, delta)
            # 选择：优先更小的 C；C 相等（或非常接近）时选更小的 E
            if cand[0] + 1e-15 < best[0] or (abs(cand[0] - best[0]) <= 1e-15 and cand[1] + improve_tol < best[1]):
                best = cand

        # 如果选了非 0 delta，真正更新
        if abs(best[2]) > 0.0:
            _ = apply_temporal_spread(dof_pos, t, d, best[2], temporal_spread)

            # 更新 base（因为后续 DOF 基于新状态继续贪心）
            C_base = collision_hinge_sum(model, data, qpos_template, dof_pos, frames_eval, coll_margin)
            x_r = get_robot_anchor_positions(model, data, qpos_template, dof_pos, t, body_ids)
            S_base = frame_semantic_direction_error(x_h, x_r, root_index)
            D_base = frame_root_relative_mpjpe(x_h, x_r, root_index)
            M_base = continuity_cost(dof_pos, t, target_dofs)
            dev_base = dev_cost(dof_pos, dof_orig, t, target_dofs)
            E_base = w_C * C_base + w_S * S_base + w_D * D_base + w_M * M_base + w_dev * dev_base


# ------------------------- main -------------------------

def process_one_pkl(input_pkl: Path,
                    output_pkl: Path,
                    model: mj.MjModel,
                    ik_cfg: dict,
                    tables: str,
                    root_index: int,
                    coll_margin: float,
                    num_passes: int,
                    delta_angle: float,
                    temporal_spread: bool,
                    w_C: float,
                    w_S: float,
                    w_D: float,
                    w_M: float,
                    w_dev: float) -> Dict:
    """
    处理单个 pkl：读取 -> 扫描 near-contact(dist < coll_margin) -> refine -> 保存。
    返回 timing dict。
    """
    t_all0 = time.perf_counter()
    timing = {
        "load_s": 0.0,
        "setup_s": 0.0,
        "scan_s": 0.0,
        "refine_s": 0.0,
        "final_scan_s": 0.0,
        "save_s": 0.0,
        "total_s": 0.0,
        "compute_s": 0.0,
        "frames": 0,
        "fps": None,
        "motion_seconds": 0.0,
        "passes_run": 0,
        "near_contact_frames_total": 0,
        "files": 1,
    }

    t0 = time.perf_counter()
    with input_pkl.open("rb") as f:
        payload = pickle.load(f)
    timing["load_s"] = time.perf_counter() - t0

    if "robot_motion" not in payload or "human_motion" not in payload:
        raise ValueError("pkl 中必须包含 robot_motion 和 human_motion")

    robot_motion = payload["robot_motion"]
    human_motion = payload["human_motion"]

    fps = robot_motion.get("fps", human_motion.get("fps", None))
    if fps is None:
        raise ValueError("无法从 pkl 中找到 fps")
    timing["fps"] = float(fps)
    print(f"[Info] fps = {fps}")

    t0 = time.perf_counter()

    # 每个文件都用独立的 MjData（避免遗留状态）
    data = mj.MjData(model)

    # qpos template and dof
    qpos_template = build_robot_qpos_template(robot_motion)
    dof_orig = np.asarray(robot_motion["dof_pos"], dtype=float)
    dof_pos = dof_orig.copy()

    T = dof_pos.shape[0]
    ndof = dof_pos.shape[1]
    print(f"[Info] T={T}, ndof={ndof}")

    # build influence dofs
    influence_dofs = build_body_influence_dofs_ordered(model, qpos_root_size=7)

    # human anchors
    root_pos_h = np.asarray(human_motion["root_pos"], dtype=float)  # (T_h,3)
    local_body_pos = np.asarray(human_motion["local_body_pos"], dtype=float)  # (T_h,Nh,3)
    human_body_names = list(human_motion["link_body_list"])

    T_h = local_body_pos.shape[0]
    human_global_all = local_body_pos + root_pos_h[:, None, :]

    # align T
    if T_h != T:
        T_align = min(T, T_h)
        print(f"[Warn] T mismatch: robot {T}, human {T_h}. Align to {T_align}")
        T = T_align
        qpos_template = qpos_template[:T]
        dof_orig = dof_orig[:T]
        dof_pos = dof_pos[:T]
        human_global_all = human_global_all[:T]
        # also truncate stored arrays for saving consistency
        robot_motion["root_pos"] = robot_motion["root_pos"][:T]
        robot_motion["root_rot"] = robot_motion["root_rot"][:T]

    # read ik_config -> pairs -> anchors
    pairs = build_pairs_from_ik_config(ik_cfg, tables=tables)
    robot_body_names = []
    human_indices = []
    for robot_body, human_name, pos_w in pairs:
        if human_name not in human_body_names:
            raise ValueError(f"Human joint {human_name} not in link_body_list")
        robot_body_names.append(robot_body)
        human_indices.append(human_body_names.index(human_name))

    J = len(robot_body_names)
    if not (0 <= root_index < J):
        raise ValueError(f"root_index={root_index} out of range [0,{J-1}]")

    body_ids = []
    for name in robot_body_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Robot body '{name}' not found in model")
        body_ids.append(bid)

    human_anchor_pos = np.zeros((T, J, 3), dtype=float)
    for j, idx_h in enumerate(human_indices):
        human_anchor_pos[:, j, :] = human_global_all[:, idx_h, :]

    print("[Info] anchor pairs:")
    for j in range(min(J, 20)):
        print(f"  {j:02d}: {robot_body_names[j]} <-> {human_body_names[human_indices[j]]}")
    print(f"[Info] root_index={root_index}: robot={robot_body_names[root_index]}")

    timing["setup_s"] = time.perf_counter() - t0
    timing["frames"] = int(T)
    timing["motion_seconds"] = float(T / fps) if fps and fps > 0 else 0.0

    # passes
    for p in range(num_passes):
        timing["passes_run"] += 1

        t0 = time.perf_counter()
        frame_contacts = collect_near_contact_frames(model, qpos_template, dof_pos, coll_margin)
        timing["scan_s"] += time.perf_counter() - t0

        frames = [t for t, cs in enumerate(frame_contacts) if cs]
        timing["near_contact_frames_total"] += len(frames)
        print(f"[Pass {p}] near-contact frames (dist < {coll_margin}): {len(frames)}/{T}")
        if not frames:
            break

        # hand-first ordering (helpful)
        hand_kw = ("hand", "wrist")
        hand_frames = []
        other_frames = []
        for t in frames:
            cs = frame_contacts[t]
            has_hand = any(any(k in c["body1_name"].lower() for k in hand_kw) or
                           any(k in c["body2_name"].lower() for k in hand_kw) for c in cs)
            (hand_frames if has_hand else other_frames).append(t)
        frames_ordered = hand_frames + other_frames

        t0 = time.perf_counter()
        for t in frames_ordered:
            refine_frame(
                model=model,
                data=data,
                qpos_template=qpos_template,
                dof_pos=dof_pos,
                dof_orig=dof_orig,
                human_anchor_pos=human_anchor_pos,
                body_ids=body_ids,
                root_index=root_index,
                frame_contacts=frame_contacts,
                influence_dofs=influence_dofs,
                t=t,
                coll_margin=coll_margin,
                delta_angle=delta_angle,
                temporal_spread=temporal_spread,
                w_C=w_C,
                w_S=w_S,
                w_D=w_D,
                w_M=w_M,
                w_dev=w_dev,
            )
        timing["refine_s"] += time.perf_counter() - t0

    # final scan
    t0 = time.perf_counter()
    final_contacts = collect_near_contact_frames(model, qpos_template, dof_pos, coll_margin)
    timing["final_scan_s"] = time.perf_counter() - t0
    final_frames = [t for t, cs in enumerate(final_contacts) if cs]
    print(f"[Final] near-contact frames (dist < {coll_margin}): {len(final_frames)}/{T}")

    # save
    t0 = time.perf_counter()
    robot_motion["dof_pos"] = dof_pos
    out = {"robot_motion": robot_motion, "human_motion": human_motion}
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pickle.dump(out, f)
    timing["save_s"] = time.perf_counter() - t0

    timing["compute_s"] = timing["scan_s"] + timing["refine_s"] + timing["final_scan_s"]
    timing["total_s"] = time.perf_counter() - t_all0

    compute_fps = (T / timing["compute_s"]) if timing["compute_s"] > 0 else float("inf")
    total_fps = (T / timing["total_s"]) if timing["total_s"] > 0 else float("inf")
    rtf = (timing["compute_s"] / timing["motion_seconds"]) if timing["motion_seconds"] > 0 else float("inf")

    print(
        f"[Speed] file={input_pkl.name} frames={T} passes={timing['passes_run']} "
        f"load={timing['load_s']:.4f}s setup={timing['setup_s']:.4f}s "
        f"scan={timing['scan_s']:.4f}s refine={timing['refine_s']:.4f}s "
        f"final_scan={timing['final_scan_s']:.4f}s save={timing['save_s']:.4f}s "
        f"compute_s={timing['compute_s']:.4f}s total_s={timing['total_s']:.4f}s "
        f"compute_fps={compute_fps:.2f} total_fps={total_fps:.2f} RTF={rtf:.4f}"
    )

    print(f"[Save] {output_pkl}")
    return timing


def main():
    parser = argparse.ArgumentParser()

    # single file mode
    parser.add_argument("--input_pkl", type=str, default=None,
                        help="单个输入 pkl。与 --input_dir 二选一。")
    parser.add_argument("--output_pkl", type=str, default=None,
                        help="单个输出 pkl（单文件模式必填）。")

    # batch mode
    parser.add_argument("--input_dir", type=str, default=None,
                        help="批处理：输入目录，遍历其中的 pkl。与 --input_pkl 二选一。")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="批处理：输出目录（默认等于 input_dir）。")
    parser.add_argument("--glob", type=str, default="*.pkl",
                        help="批处理：匹配模式（默认 *.pkl）。")
    parser.add_argument("--recursive", action="store_true",
                        help="批处理：递归遍历子目录（使用 rglob）。")
    parser.add_argument("--keep_subdirs", action="store_true",
                        help="批处理：递归时保留子目录结构到 output_dir（默认开启）。")
    parser.add_argument("--no_keep_subdirs", action="store_true",
                        help="批处理：递归时不保留子目录结构，全部平铺到 output_dir。")
    parser.add_argument("--out_suffix", type=str, default="_refine",
                        help="批处理：输出文件名后缀（默认 _refine）。")
    parser.add_argument("--overwrite", action="store_true",
                        help="批处理：若输出已存在则覆盖（默认跳过）。")
    parser.add_argument("--skip_errors", action="store_true",
                        help="批处理：单个文件报错时跳过并继续处理其余文件。")
    parser.add_argument("--save_speed_json", type=str, default=None,
                        help="可选：将测速结果保存为 json。")

    # shared configs
    parser.add_argument("--robot_xml", type=str, required=True)
    parser.add_argument("--ik_config", type=str, required=True)
    parser.add_argument("--tables", type=str, default="both", choices=["1", "2", "both"])
    parser.add_argument("--root_index", type=int, default=0)

    # hinge margin (meters)
    parser.add_argument("--coll_margin", type=float, default=0.002,
                        help="安全裕量 m（米）。dist < m 会被惩罚/视为需修复。默认 0.002=2mm")

    # passes and step
    parser.add_argument("--num_passes", type=int, default=2,
                        help="全局扫描+修复轮数。默认 2（精修常用 1~3）")
    parser.add_argument("--delta_angle", type=float, default=0.03,
                        help="每次试探的角度步长（弧度）。精修建议 0.01~0.02")

    # continuity help
    parser.add_argument("--temporal_spread", action="store_true",
                        help="将一次更新扩散到 t-1/t/t+1（更丝滑），默认开启")
    parser.add_argument("--no_temporal_spread", action="store_true",
                        help="关闭 temporal_spread")

    # weights
    parser.add_argument("--w_C", type=float, default=15.0)
    parser.add_argument("--w_S", type=float, default=1.0)
    parser.add_argument("--w_D", type=float, default=0.3)
    parser.add_argument("--w_M", type=float, default=3.0)
    parser.add_argument("--w_dev", type=float, default=1.0)

    args = parser.parse_args()

    # temporal spread default = True
    temporal_spread = True
    if args.no_temporal_spread:
        temporal_spread = False
    if args.temporal_spread:
        temporal_spread = True

    # mode check
    if (args.input_pkl is None) == (args.input_dir is None):
        raise ValueError("必须且只能指定一个：--input_pkl 或 --input_dir")

    if args.input_pkl is not None:
        if args.output_pkl is None:
            raise ValueError("单文件模式需要 --output_pkl")
    else:
        # batch mode
        if args.output_dir is None:
            args.output_dir = args.input_dir

    # load mujoco model once
    model = mj.MjModel.from_xml_path(args.robot_xml)

    # load ik config once
    with open(args.ik_config, "r", encoding="utf-8") as f:
        ik_cfg = json.load(f)

    if args.input_pkl is not None:
        input_pkl = Path(args.input_pkl)
        output_pkl = Path(args.output_pkl)
        timing = process_one_pkl(
            input_pkl=input_pkl,
            output_pkl=output_pkl,
            model=model,
            ik_cfg=ik_cfg,
            tables=args.tables,
            root_index=args.root_index,
            coll_margin=args.coll_margin,
            num_passes=args.num_passes,
            delta_angle=args.delta_angle,
            temporal_spread=temporal_spread,
            w_C=args.w_C, w_S=args.w_S, w_D=args.w_D, w_M=args.w_M, w_dev=args.w_dev,
        )
        compute_fps = (timing["frames"] / timing["compute_s"]) if timing["compute_s"] > 0 else float("inf")
        total_fps = (timing["frames"] / timing["total_s"]) if timing["total_s"] > 0 else float("inf")
        rtf = (timing["compute_s"] / timing["motion_seconds"]) if timing["motion_seconds"] > 0 else float("inf")
        print(
            f"[SpeedSummary] files=1 frames={timing['frames']} compute_s={timing['compute_s']:.4f}s total_s={timing['total_s']:.4f}s "
            f"avg_compute_fps={compute_fps:.2f} avg_total_fps={total_fps:.2f} avg_RTF={rtf:.4f}"
        )
        if args.save_speed_json:
            import json as _json
            with open(args.save_speed_json, "w", encoding="utf-8") as f:
                _json.dump({"summary": timing}, f, indent=2, ensure_ascii=False)
        return

    # batch
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_subdirs = True
    if args.no_keep_subdirs:
        keep_subdirs = False
    if args.keep_subdirs:
        keep_subdirs = True

    if args.recursive:
        in_files = sorted(in_dir.rglob(args.glob))
    else:
        in_files = sorted(in_dir.glob(args.glob))

    if not in_files:
        print(f"[Warn] No files matched: dir={in_dir} glob={args.glob} recursive={args.recursive}")
        return

    print(f"[Batch] Found {len(in_files)} files in {in_dir} (glob={args.glob}, recursive={args.recursive})")

    ok = 0
    skipped = 0
    failed = 0
    speed_records = []
    speed_summary = {
        "files": 0,
        "frames": 0,
        "load_s": 0.0,
        "setup_s": 0.0,
        "scan_s": 0.0,
        "refine_s": 0.0,
        "final_scan_s": 0.0,
        "save_s": 0.0,
        "compute_s": 0.0,
        "total_s": 0.0,
        "motion_seconds": 0.0,
    }

    for i, in_pkl in enumerate(in_files, 1):
        # output path
        if args.recursive and keep_subdirs:
            rel = in_pkl.relative_to(in_dir)
            out_parent = out_dir / rel.parent
        else:
            out_parent = out_dir

        out_name = in_pkl.stem + args.out_suffix + in_pkl.suffix
        out_pkl = out_parent / out_name

        if out_pkl.exists() and not args.overwrite:
            print(f"[{i}/{len(in_files)}] Skip (exists): {out_pkl}")
            skipped += 1
            continue

        print(f"\n[{i}/{len(in_files)}] Processing: {in_pkl}")
        try:
            timing = process_one_pkl(
                input_pkl=in_pkl,
                output_pkl=out_pkl,
                model=model,
                ik_cfg=ik_cfg,
                tables=args.tables,
                root_index=args.root_index,
                coll_margin=args.coll_margin,
                num_passes=args.num_passes,
                delta_angle=args.delta_angle,
                temporal_spread=temporal_spread,
                w_C=args.w_C, w_S=args.w_S, w_D=args.w_D, w_M=args.w_M, w_dev=args.w_dev,
            )
            speed_records.append({"file": str(in_pkl), **timing})
            speed_summary["files"] += 1
            speed_summary["frames"] += timing["frames"]
            speed_summary["load_s"] += timing["load_s"]
            speed_summary["setup_s"] += timing["setup_s"]
            speed_summary["scan_s"] += timing["scan_s"]
            speed_summary["refine_s"] += timing["refine_s"]
            speed_summary["final_scan_s"] += timing["final_scan_s"]
            speed_summary["save_s"] += timing["save_s"]
            speed_summary["compute_s"] += timing["compute_s"]
            speed_summary["total_s"] += timing["total_s"]
            speed_summary["motion_seconds"] += timing["motion_seconds"]
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[Error] {in_pkl}: {type(e).__name__}: {e}")
            if not args.skip_errors:
                raise

    print(f"\n[Batch Done] ok={ok}, skipped={skipped}, failed={failed}, out_dir={out_dir}")
    if speed_summary["frames"] > 0:
        avg_compute_fps = speed_summary["frames"] / speed_summary["compute_s"] if speed_summary["compute_s"] > 0 else float("inf")
        avg_total_fps = speed_summary["frames"] / speed_summary["total_s"] if speed_summary["total_s"] > 0 else float("inf")
        avg_rtf = speed_summary["compute_s"] / speed_summary["motion_seconds"] if speed_summary["motion_seconds"] > 0 else float("inf")
        print(
            f"[SpeedSummary] files={speed_summary['files']} frames={speed_summary['frames']} "
            f"load_s={speed_summary['load_s']:.4f}s setup_s={speed_summary['setup_s']:.4f}s "
            f"scan_s={speed_summary['scan_s']:.4f}s refine_s={speed_summary['refine_s']:.4f}s "
            f"final_scan_s={speed_summary['final_scan_s']:.4f}s save_s={speed_summary['save_s']:.4f}s "
            f"compute_s={speed_summary['compute_s']:.4f}s total_s={speed_summary['total_s']:.4f}s "
            f"avg_compute_fps={avg_compute_fps:.2f} avg_total_fps={avg_total_fps:.2f} avg_RTF={avg_rtf:.4f}"
        )
        if args.save_speed_json:
            import json as _json
            with open(args.save_speed_json, "w", encoding="utf-8") as f:
                _json.dump({"summary": speed_summary, "files": speed_records}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()


"""  
  python tools/refine_collision_minchange_hinge_batch_timed.py \
  --input_dir retargeting_result/g1_test_gmr \
  --output_dir retargeting_result/g1_test_gmr_refine_test_time \
  --glob "*.pkl" \
  --robot_xml /home/nubot/GMR_old/assets/unitree_g1/g1_mocap_29dof.xml \
  --ik_config general_motion_retargeting/ik_configs/bvh_to_g1.json \
  --tables both \
  --coll_margin 0.002 \
  --num_passes 2 \
  --delta_angle 0.03 \
  --save_speed_json retargeting_result/g1_test_gmr_refine_test_time/speed.json

  python tools/refine_collision_minchange_hinge_batch.py \
  --input_pkl  retargeting_result/lafan1_g1_save/aiming1_subject4_c.pkl \
  --output_pkl retargeting_result/lafan1_g1_save/aiming1_subject4_refine.pkl \
  --robot_xml  assets/unitree_g1/g1_mocap_29dof.xml \
  --ik_config  general_motion_retargeting/ik_configs/bvh_to_g1.json

"""