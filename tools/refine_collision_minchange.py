#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
refine_collision_minchange.py

思路：
  - 只处理“有碰撞的帧”；
  - 对每个碰撞帧 t：
      * 找出参与碰撞的 body（若有 hand/wrist，则优先只用那条手臂链条）；
      * 用 body->DOF 影响关系，得到一组 target_dofs（这些 DOF 可以动来消碰撞）；
      * 对每个 target_dof，局部试探三个候选：0, +delta, -delta（极小改动）；
      * 每个候选都计算一个局部代价：
            E = w_C * C_coll    # 碰撞穿透
              + w_S * S_sem     # 方向语义差（手在头上这种关系）
              + w_D * D_mpjpe   # root-relative MPJPE
              + w_M * M_smooth  # 同一关节与前后帧的连续性
              + w_dev * Dev     # 偏离原始 GMR 的大小
        在候选里选 cost 最小的，并且 !优先考虑 C_coll 降低/归零!。

  - 可以设定全局 pass 次数，对所有碰撞帧迭代几轮；
  - 只改动“碰撞帧 + 碰撞相关 DOFs”，其余时间与关节几乎不动；
  - 不再使用随机方向 / SPSA，因此不会产生随机抖动。

用法示例：

  python tools/refine_collision_minchange.py \
    --input_pkl  retargeting_result/lafan1_g1/aiming1_subject1_c.pkl \
    --robot_xml  assets/unitree_g1/g1_mocap_29dof.xml \
    --ik_config  general_motion_retargeting/ik_configs/bvh_to_g1.json \
    --tables     both \
    --output_pkl retargeting_result/lafan1_g1/aiming1_subject1_minchange.pkl
"""

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import mujoco as mj


# ===================== 通用：qpos / DOF / 关节对 =====================

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
        raise ValueError("在 ik_match_table 中没有找到 pos_weight > 0 的关节，无法构造评估关节对。")

    return pairs


def build_body_influence_dofs(model, qpos_root_size=7):
    """
    对于每个 body，给出所有能影响它的 DOF 索引（在 dof_pos 里的列索引）。

    做法：
      - 从 body b 一路沿 parent 链走到 world；
      - 每个路过的 body 上的 joint，其 qpos 维度映射到 dof_pos 的索引：
            dof_idx = qpos_idx - qpos_root_size
    """
    nbody = model.nbody
    njnt = model.njnt
    nq = model.nq

    body_parent = model.body_parentid
    body_jntnum = model.body_jntnum
    body_jntadr = model.body_jntadr
    jnt_qposadr = model.jnt_qposadr

    num_qpos_per_jnt = np.zeros(njnt, dtype=int)
    for j in range(njnt):
        start = jnt_qposadr[j]
        end = jnt_qposadr[j + 1] if j + 1 < njnt else nq
        num_qpos_per_jnt[j] = end - start

    influence = [set() for _ in range(nbody)]

    for b in range(nbody):
        cur = b
        while cur > 0:  # 0 = world
            jnum = body_jntnum[cur]
            jadr = body_jntadr[cur]
            for k in range(jnum):
                j = jadr + k
                start = jnt_qposadr[j]
                end = start + num_qpos_per_jnt[j]
                for q_idx in range(start, end):
                    if q_idx >= qpos_root_size:
                        dof_idx = q_idx - qpos_root_size
                        influence[b].add(dof_idx)
            cur = body_parent[cur]

    return [sorted(list(s)) for s in influence]


# ===================== 碰撞扫描 =====================

def collect_collision_info(model, qpos_template, dof_pos):
    """
    整条轨迹扫描自碰撞。

    qpos_template: (T, nq)，其中前 7 维为 root，7: 将由 dof_pos 覆盖。
    dof_pos:       (T, ndof)

    返回：
      frame_contacts: list[T]，每个元素是 list[dict]，包含：
          {
            "body1": int,
            "body2": int,
            "body1_name": str,
            "body2_name": str,
            "dist": float,   # dist < 0 表示穿透
          }
    """
    T = dof_pos.shape[0]
    frame_contacts = [[] for _ in range(T)]
    data = mj.MjData(model)

    for t in range(T):
        qpos = qpos_template[t].copy()
        qpos[7:] = dof_pos[t]
        data.qpos[:] = qpos
        mj.mj_forward(model, data)

        for ci in range(data.ncon):
            contact = data.contact[ci]
            g1 = contact.geom1
            g2 = contact.geom2
            if g1 < 0 or g2 < 0:
                continue

            b1 = model.geom_bodyid[g1]
            b2 = model.geom_bodyid[g2]
            name1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b1)
            name2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b2)

            if name1 == "world" or name2 == "world":
                continue

            if contact.dist <  0.0:
                frame_contacts[t].append({
                    "body1": int(b1),
                    "body2": int(b2),
                    "body1_name": name1,
                    "body2_name": name2,
                    "dist": float(contact.dist),
                })

    return frame_contacts


def get_target_dofs_for_frame(t, frame_contacts, influence_dofs,
                              hand_keywords=("hand", "wrist")):
    """
    对于帧 t，根据碰撞信息确定“可调整的 DOF 集合”：

      - 若有 body 名包含 hand/wrist，则只用这些 hand/wrist body；
      - 否则用所有碰撞涉及的 body；
      - 对这些 body，在 influence_dofs 里查出能影响它们的所有 DOF。
    """
    contacts = frame_contacts[t]
    if not contacts:
        return []

    hand_bodies = set()
    other_bodies = set()
    for c in contacts:
        for bid, name in [(c["body1"], c["body1_name"]),
                          (c["body2"], c["body2_name"])]:
            lname = name.lower()
            if any(k in lname for k in hand_keywords):
                hand_bodies.add(bid)
            else:
                other_bodies.add(bid)

    if hand_bodies:
        bodies = hand_bodies
    else:
        bodies = hand_bodies.union(other_bodies)

    target_dofs = set()
    for b in bodies:
        for d in influence_dofs[b]:
            target_dofs.add(d)

    return sorted(list(target_dofs))


# ===================== 单帧评估：C + S + D + M + dev =====================

def evaluate_frame_cost(model,
                        data,
                        qpos_template,
                        dof_pos,
                        dof_orig,
                        human_anchor_pos,
                        body_ids,
                        root_index,
                        t,
                        w_C=10.0,
                        w_S=1.0,
                        w_D=0.5,
                        w_M=1.0,
                        w_dev=0.5):
    """
    评估单帧 t 的局部代价：

      C: 自碰撞穿透深度平方和（只看这一帧）；
      S: 语义方向误差（pelvis->各 anchor 单位向量的 1 - cosθ）；
      D: root-relative MPJPE（这一帧）；
      M: 关节连续性（与前后帧的 L2 差）；
      dev: 与原始 GMR dof 的偏离。

    返回：
      E_total, metrics_dict
        metrics_dict = {"C":..., "S":..., "D":..., "M":..., "dev":...}
    """
    T, ndof = dof_pos.shape
    J = len(body_ids)

    # ---------- 1. 自碰撞 C ----------
    qpos = qpos_template[t].copy()
    qpos[7:] = dof_pos[t]
    data.qpos[:] = qpos
    mj.mj_forward(model, data)

    coll_acc = 0.0
    for ci in range(data.ncon):
        contact = data.contact[ci]
        g1 = contact.geom1
        g2 = contact.geom2
        if g1 < 0 or g2 < 0:
            continue
        b1 = model.geom_bodyid[g1]
        b2 = model.geom_bodyid[g2]
        name1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b1)
        name2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b2)
        if name1 == "world" or name2 == "world":
            continue
        if contact.dist < 0.0:
            depth = -float(contact.dist)
            coll_acc += depth * depth
    C = coll_acc

    # ---------- 2. 机器人 anchor 位置 ----------
    x_robot = np.zeros((J, 3), dtype=float)
    for j, bid in enumerate(body_ids):
        x_robot[j] = data.xpos[bid]

    x_human = human_anchor_pos[t]  # (J,3)

    # ---------- 3. S: 语义方向误差 ----------
    xh_root = x_human[root_index]
    xr_root = x_robot[root_index]
    vec_h = x_human - xh_root
    vec_r = x_robot - xr_root

    eps = 1e-8
    nh = vec_h / (np.linalg.norm(vec_h, axis=-1, keepdims=True) + eps)
    nr = vec_r / (np.linalg.norm(vec_r, axis=-1, keepdims=True) + eps)

    dot = np.clip(np.sum(nh * nr, axis=-1), -1.0, 1.0)  # (J,)
    S_err = float(np.mean(1.0 - dot))   # 0 最好

    # ---------- 4. D: root-relative MPJPE（这一帧） ----------
    rel_h = vec_h
    rel_r = vec_r
    diff = rel_r - rel_h
    D_err = float(np.mean(np.linalg.norm(diff, axis=-1)))  # 尺度：米

    # ---------- 5. M: 与前后帧的关节连续性 ----------
    M_err = 0.0
    cnt = 0
    if t > 0:
        diff_prev = dof_pos[t] - dof_pos[t - 1]
        M_err += float(np.mean(diff_prev ** 2))
        cnt += 1
    if t < T - 1:
        diff_next = dof_pos[t + 1] - dof_pos[t]
        M_err += float(np.mean(diff_next ** 2))
        cnt += 1
    if cnt > 0:
        M_err /= cnt

    # ---------- 6. dev: 偏离原始 GMR ----------
    dev = dof_pos[t] - dof_orig[t]
    dev_err = float(np.mean(dev ** 2))

    E_total = (
        w_C * C +
        w_S * S_err +
        w_D * D_err +
        w_M * M_err +
        w_dev * dev_err
    )

    metrics = {"C": C, "S": S_err, "D": D_err, "M": M_err, "dev": dev_err}
    return E_total, metrics


# ===================== 单帧最小改动修复 =====================

def refine_frame_minchange(model,
                           data,
                           qpos_template,
                           dof_pos,
                           dof_orig,
                           human_anchor_pos,
                           body_ids,
                           root_index,
                           frame_contacts,
                           influence_dofs,
                           t,
                           delta_angle=0.05,
                           w_C=10.0,
                           w_S=1.0,
                           w_D=0.5,
                           w_M=1.0,
                           w_dev=0.5,
                           improve_tol=1e-6):
    """
    对单帧 t 做“最小改动消碰撞”的局部搜索：

      1. 如果该帧没有碰撞，直接返回；
      2. 用 get_target_dofs_for_frame 得到 target_dofs（手部优先）；
      3. 对每个 target_dof，试探 delta ∈ {0, +δ, -δ}：
          - 计算该帧的 E_total；
          - 优先选 C 降低/归零的候选；
      4. 若有候选能显著降低 E_total 或 C，就更新该 DOF。

    注意：
      - 这是一个“贪心 + 极小扰动”的过程；
      - dof_pos 是全局数组，在这里就地修改 frame t。
    """
    contacts_t = frame_contacts[t]
    if not contacts_t:
        return dof_pos

    target_dofs = get_target_dofs_for_frame(t, frame_contacts, influence_dofs)
    if not target_dofs:
        return dof_pos

    # 原始 cost（不动任何 DOF）
    base_E, base_metrics = evaluate_frame_cost(
        model, data, qpos_template, dof_pos, dof_orig,
        human_anchor_pos, body_ids, root_index, t,
        w_C=w_C, w_S=w_S, w_D=w_D, w_M=w_M, w_dev=w_dev
    )

    # 逐个 DOF 做局部贪心搜索
    for d in target_dofs:
        best_E = base_E
        best_metrics = base_metrics
        best_delta = 0.0

        # 尝试三个候选：0, +δ, -δ
        for delta in (0.0, +delta_angle, -delta_angle):
            if delta == 0.0:
                # 原值，已经算过
                E, metrics = base_E, base_metrics
            else:
                # 暂时修改
                old_val = dof_pos[t, d]
                dof_pos[t, d] = old_val + delta

                E, metrics = evaluate_frame_cost(
                    model, data, qpos_template, dof_pos, dof_orig,
                    human_anchor_pos, body_ids, root_index, t,
                    w_C=w_C, w_S=w_S, w_D=w_D, w_M=w_M, w_dev=w_dev
                )

                # 恢复
                dof_pos[t, d] = old_val

            # 选择策略：
            #  - 若碰撞 C 明显下降（特别是从 >0 -> 0），优先；
            #  - 在 C 差不多时，用 E_total 比较。
            C_base = best_metrics["C"]
            C_now = metrics["C"]

            better = False
            if C_base > 1e-8 and C_now < C_base - 1e-8:
                # 碰撞更少（穿透减小），直接认为更好
                better = True
            elif abs(C_now - C_base) < 1e-8:
                # 碰撞类似，则比较总 cost
                if E < best_E - improve_tol:
                    better = True
            else:
                # C_now > C_base：通常不接受，除非极大降低了 S/D/M/dev
                # 这里保守起见：不接受
                better = False

            if better:
                best_E = E
                best_metrics = metrics
                best_delta = delta

        # 如果这个 DOF 有收益，就更新
        if abs(best_delta) > 0.0:
            dof_pos[t, d] += best_delta
            base_E = best_E
            base_metrics = best_metrics

    return dof_pos


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pkl", type=str, required=True,
                        help="包含 robot_motion & human_motion 的输入 pkl")
    parser.add_argument("--robot_xml", type=str, required=True,
                        help="MuJoCo 机器人 xml（例如 g1_mocap_29dof.xml）")
    parser.add_argument("--ik_config", type=str, required=True,
                        help="IK 配置 json（例如 bvh_to_g1.json）")
    parser.add_argument("--tables", type=str, default="both",
                        choices=["1", "2", "both"],
                        help="使用 ik_match_table1 / 2 / both 中的哪些关节（默认 both）")
    parser.add_argument("--output_pkl", type=str, required=True,
                        help="保存优化后结果的 pkl 路径")

    parser.add_argument("--root_index", type=int, default=0,
                        help="评估关节对中哪个 index 作为 root（默认 0，对应 pelvis-Hips）")

    parser.add_argument("--num_passes", type=int, default=2,
                        help="全局遍历碰撞帧的次数（默认 2）")
    parser.add_argument("--delta_angle", type=float, default=0.05,
                        help="每次试探的关节弧度改变量（默认 0.05 ≈ 2.8°）")

    parser.add_argument("--w_C", type=float, default=10.0,
                        help="碰撞项权重（默认 10.0）")
    parser.add_argument("--w_S", type=float, default=1.0,
                        help="语义方向项权重（默认 1.0）")
    parser.add_argument("--w_D", type=float, default=0.5,
                        help="MPJPE 项权重（默认 0.5）")
    parser.add_argument("--w_M", type=float, default=1.0,
                        help="关节连续性项权重（默认 1.0）")
    parser.add_argument("--w_dev", type=float, default=0.5,
                        help="偏离原始 GMR 的正则权重（默认 0.5）")
    args = parser.parse_args()

    input_pkl = Path(args.input_pkl)
    if not input_pkl.exists():
        raise FileNotFoundError(input_pkl)

    with input_pkl.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError("input pkl 顶层不是 dict。")
    if "robot_motion" not in data or "human_motion" not in data:
        raise ValueError("input pkl 缺少 robot_motion 或 human_motion。")

    robot_motion = data["robot_motion"]
    human_motion = data["human_motion"]

    # fps
    fps_r = robot_motion.get("fps", None)
    fps_h = human_motion.get("fps", None)
    if fps_r is None and fps_h is None:
        raise ValueError("无法从 pkl 中找到 fps 字段。")
    fps = fps_r if fps_r is not None else fps_h
    print(f"[Info] 使用 fps = {fps} Hz")

    # MuJoCo 模型
    model = mj.MjModel.from_xml_path(args.robot_xml)
    data_mj = mj.MjData(model)

    # qpos 模板 & DOFs
    qpos_template = build_robot_qpos(robot_motion)   # (T, nq)
    T, nq = qpos_template.shape
    ndof = nq - 7

    dof_orig = np.asarray(robot_motion["dof_pos"], dtype=float)
    assert dof_orig.shape == (T, ndof)
    dof_pos = dof_orig.copy()

    # body->DOF 映射
    influence_dofs = build_body_influence_dofs(model, qpos_root_size=7)
    print(f"[Info] 已构建 body_influence_dofs，共 {len(influence_dofs)} 个 body。")

    # human anchor 坐标（根据 ik_config + tables 选取）
    root_pos_h = np.asarray(human_motion["root_pos"], dtype=float)           # (T_h,3)
    local_body_pos = np.asarray(human_motion["local_body_pos"], dtype=float) # (T_h,Nh,3)
    human_body_names = list(human_motion["link_body_list"])                  # 长度 Nh

    T_h, Nh, _ = local_body_pos.shape
    human_global_all = local_body_pos + root_pos_h[:, None, :]               # (T_h,Nh,3)

    # 对齐长度（一般 T_h == T）
    if T_h != T:
        T_align = min(T, T_h)
        print(f"[Info] robot T={T}, human T={T_h}，对齐到 T={T_align}")
        T = T_align
        qpos_template = qpos_template[:T]
        dof_orig = dof_orig[:T]
        dof_pos = dof_pos[:T]
        human_global_all = human_global_all[:T]
        robot_motion["root_pos"] = robot_motion["root_pos"][:T]
        robot_motion["root_rot"] = robot_motion["root_rot"][:T]
        robot_motion["dof_pos"] = robot_motion["dof_pos"][:T]

    # 读取 ik_config，构造评估关节对
    with open(args.ik_config, "r", encoding="utf-8") as f:
        ik_cfg = json.load(f)
    pairs = build_pairs_from_ik_config(ik_cfg, tables=args.tables)

    robot_body_names = []
    human_indices = []
    for robot_body_name, human_name, pos_w in pairs:
        if human_name not in human_body_names:
            raise ValueError(f"Human 关节 {human_name} 不在 human_motion['link_body_list'] 中。")
        idx_h = human_body_names.index(human_name)
        robot_body_names.append(robot_body_name)
        human_indices.append(idx_h)

    J = len(robot_body_names)
    print("评估关节对（robot ↔ human）：")
    for j in range(J):
        print(f"  {j:02d}: {robot_body_names[j]}  <->  {human_body_names[human_indices[j]]}")
    print(f"Root pair: index={args.root_index}, robot={robot_body_names[args.root_index]}, human={human_body_names[human_indices[args.root_index]]}")

    # anchor 对应的 body id
    body_ids = []
    for name in robot_body_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in model.")
        body_ids.append(bid)

    # 构造 human_anchor_pos: (T, J, 3)
    human_anchor_pos = np.zeros((T, J, 3), dtype=float)
    for j, idx_h in enumerate(human_indices):
        human_anchor_pos[:, j, :] = human_global_all[:, idx_h, :]

    # ---------------- 全局多 pass 的局部修复 ----------------
    for p in range(args.num_passes):
        print(f"\n[Pass {p}] 扫描碰撞...")
        frame_contacts = collect_collision_info(model, qpos_template, dof_pos)
        frames_with_collision = [t for t, cs in enumerate(frame_contacts) if cs]
        num_coll = len(frames_with_collision)
        print(f"[Pass {p}] 当前有碰撞的帧数：{num_coll} / {T}")

        if num_coll == 0:
            print(f"[Pass {p}] 无碰撞，提前结束。")
            break

        # 对每个有碰撞的帧做局部最小改动搜索
        for t in frames_with_collision:
            dof_pos = refine_frame_minchange(
                model=model,
                data=data_mj,
                qpos_template=qpos_template,
                dof_pos=dof_pos,
                dof_orig=dof_orig,
                human_anchor_pos=human_anchor_pos,
                body_ids=body_ids,
                root_index=args.root_index,
                frame_contacts=frame_contacts,
                influence_dofs=influence_dofs,
                t=t,
                delta_angle=args.delta_angle,
                w_C=args.w_C,
                w_S=args.w_S,
                w_D=args.w_D,
                w_M=args.w_M,
                w_dev=args.w_dev,
            )

    # 最终一次碰撞统计
    print("\n[Final] 再次扫描碰撞情况...")
    frame_contacts_final = collect_collision_info(model, qpos_template, dof_pos)
    frames_with_collision_final = [t for t, cs in enumerate(frame_contacts_final) if cs]
    print(f"[Final] 碰撞帧数：{len(frames_with_collision_final)} / {T}")

    # 保存结果（只更新 dof_pos）
    robot_motion["dof_pos"] = dof_pos
    save_data = {
        "robot_motion": robot_motion,
        "human_motion": human_motion,
    }

    output_pkl = Path(args.output_pkl)
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pickle.dump(save_data, f)

    print(f"[Save] 已保存到 {output_pkl}")


if __name__ == "__main__":
    main()
