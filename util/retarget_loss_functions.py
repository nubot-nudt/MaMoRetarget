# net/retarget_loss_functions.py
# -*- coding: utf-8 -*-

"""
retarget_loss_functions.py

目标：把“语义处理/可微FK相关逻辑”集中在这里，训练脚本只做简单调用。

提供：
1) RobotTorchFK：torch 可微树形 FK（用于语义 loss 的反传）
2) jerk_mse：vel/acc/jerk 阶差分 MSE（可做平滑正则，或与 GT jerk 对齐）
3) semantic_direction_loss：参考 refine_collision_minchange_hinge.py 的方向语义 (mean(1-cos))
4) SemanticLossComputer：封装“读取 ik_config -> 构建 anchors -> human/robot anchors -> 语义loss”全过程

并提供 CLI main()：
- 读取包含 {"robot_motion","human_motion"} 的 pkl
- 用 torch FK 计算 anchors positions
- (可选) 用 MuJoCo 做 FK 对比验证（支持 body_xpos / joint_xanchor 两种对齐方式）
- 打印：torchFK vs mujocoFK 误差、semantic_direction_error 等

运行示例：
python net/util/retarget_loss_functions.py \
  --pkl retargeting_result/g1/net_out_pkls/aiming2_subject5.pkl \
  --robot_spec_npz net/worknet/robot_spec_g1.npz \
  --ik_config general_motion_retargeting/ik_configs/bvh_to_g1.json \
  --tables both \
  --root_robot pelvis \
  --robot_xml assets/unitree_g1/g1_mocap_29dof.xml \
  --mujoco_point joint_xanchor
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Quaternion utils (wxyz)
# ============================================================

def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
    # q = (w,x,y,z)
    w, x, y, z = q.unbind(dim=-1)
    return torch.stack([w, -x, -y, -z], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # Hamilton product, wxyz
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector v by quaternion q (wxyz).
    q: (...,4), v: (...,3)
    """
    q = quat_normalize(q)
    v_as_quat = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)  # (0, vx, vy, vz)
    return quat_mul(quat_mul(q, v_as_quat), quat_conj(q))[..., 1:]


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    axis: (...,3), angle: (...,)
    return: (...,4) wxyz
    """
    axis_n = axis / (axis.norm(dim=-1, keepdim=True) + eps)
    half = 0.5 * angle
    w = torch.cos(half)
    s = torch.sin(half)
    xyz = axis_n * s.unsqueeze(-1)
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)


def ensure_wxyz(q: torch.Tensor, quat_order: str) -> torch.Tensor:
    """
    q: (...,4)
    quat_order:
      - "xyzw": input is (x,y,z,w)
      - "wxyz": input is (w,x,y,z)
    """
    if quat_order == "wxyz":
        return q
    if quat_order == "xyzw":
        x, y, z, w = q.unbind(dim=-1)
        return torch.stack([w, x, y, z], dim=-1)
    raise ValueError(f"Unknown quat_order={quat_order}")


# ============================================================
# finite diff / jerk
# ============================================================

def _finite_diff(x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    x: (B,T,D) -> diff along T.
    order=1: v[t]=x[t+1]-x[t]
    order=2: a[t]=v[t+1]-v[t]
    order=3: j[t]=a[t+1]-a[t]
    """
    y = x
    for _ in range(order):
        y = y[:, 1:, :] - y[:, :-1, :]
    return y


def jerk_mse(pred: torch.Tensor,
             target: Optional[torch.Tensor] = None,
             mode: str = "jerk",
             reduction: str = "mean") -> torch.Tensor:
    """
    pred/target: (B,T,D)  (或 (T,D) 也行，会自动补 batch)
    mode:
      - "vel":  1阶差分匹配
      - "acc":  2阶差分匹配
      - "jerk": 3阶差分匹配
    若 target=None，则返回 pred 的差分能量（鼓励平滑）。
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    if target is not None and target.dim() == 2:
        target = target.unsqueeze(0)

    order = {"vel": 1, "acc": 2, "jerk": 3}.get(mode, None)
    if order is None:
        raise ValueError(f"Unknown mode={mode}, choose from vel/acc/jerk")

    dp = _finite_diff(pred, order=order)
    if target is None:
        loss = (dp ** 2)
    else:
        dt = _finite_diff(target, order=order)
        loss = (dp - dt) ** 2

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


# ============================================================
# semantic losses (refine_collision_minchange_hinge.py style)
# ============================================================

def semantic_direction_loss(x_h: torch.Tensor,
                            x_r: torch.Tensor,
                            root_index: int,
                            weights: Optional[torch.Tensor] = None,
                            eps: float = 1e-8) -> torch.Tensor:
    """
    x_h, x_r: (B,T,J,3)
    loss = mean_j (1 - cos(theta_j))  (theta between root-relative vectors)

    注：这是“方向语义”，对平移不敏感（因为 root-relative）。
    """
    vh = x_h - x_h[:, :, root_index:root_index+1, :]
    vr = x_r - x_r[:, :, root_index:root_index+1, :]

    nh = vh / (vh.norm(dim=-1, keepdim=True) + eps)
    nr = vr / (vr.norm(dim=-1, keepdim=True) + eps)

    dot = (nh * nr).sum(dim=-1).clamp(-1.0, 1.0)  # (B,T,J)
    per = 1.0 - dot

    if weights is not None:
        w = weights.view(1, 1, -1).to(per.device, per.dtype)
        per = per * w
        denom = w.sum().clamp_min(eps)
        return per.sum() / denom / (per.shape[0] * per.shape[1])
    return per.mean()


# ============================================================
# IK config -> anchor pairs (same logic as refine_collision_minchange_hinge.py)
# ============================================================

def build_pairs_from_ik_config(ik_cfg: dict, tables: str = "both") -> List[Tuple[str, str, float]]:
    """
    返回：[(robot_body_name, human_name, pos_weight), ...]
    只保留 pos_weight >= 0 (table1) / >0 (table2) 的项（与 refine 脚本一致）
    """
    use_table1 = tables in ("1", "both") and ik_cfg.get("use_ik_match_table1", True)
    use_table2 = tables in ("2", "both") and ik_cfg.get("use_ik_match_table2", True)

    pairs: List[Tuple[str, str, float]] = []
    seen = set()

    if use_table1 and "ik_match_table1" in ik_cfg:
        for robot_body_name, entry in ik_cfg["ik_match_table1"].items():
            human_name, pos_w = entry[0], float(entry[1])
            if pos_w >= 0:
                key = (robot_body_name, human_name)
                if key not in seen:
                    pairs.append((robot_body_name, human_name, pos_w))
                    seen.add(key)

    if use_table2 and "ik_match_table2" in ik_cfg:
        for robot_body_name, entry in ik_cfg["ik_match_table2"].items():
            human_name, pos_w = entry[0], float(entry[1])
            if pos_w > 0:
                key = (robot_body_name, human_name)
                if key not in seen:
                    pairs.append((robot_body_name, human_name, pos_w))
                    seen.add(key)

    if not pairs:
        raise ValueError("在 ik_match_table 中没有找到 pos_weight>0 的关节对。")
    return pairs


# ============================================================
# motion extraction helpers
# ============================================================

def human_global_anchors_from_motion(human_motion: dict,
                                    human_anchor_names: Sequence[str]) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    human_motion:
      - root_pos: (T,3)
      - local_body_pos: (T,Nh,3)
      - link_body_list: [Nh]
    按 refine 的做法：human_global = local_body_pos + root_pos[:,None,:]
    """
    root_pos_h = np.asarray(human_motion["root_pos"], dtype=float)
    local_body_pos = np.asarray(human_motion["local_body_pos"], dtype=float)
    human_body_names = list(human_motion["link_body_list"])

    human_global_all = local_body_pos + root_pos_h[:, None, :]

    human_indices: List[int] = []
    for name in human_anchor_names:
        if name not in human_body_names:
            raise ValueError(f"Human joint '{name}' not in link_body_list")
        human_indices.append(human_body_names.index(name))

    T = human_global_all.shape[0]
    J = len(human_anchor_names)
    out = np.zeros((T, J, 3), dtype=float)
    for j, idx in enumerate(human_indices):
        out[:, j, :] = human_global_all[:, idx, :]
    return out, human_indices, human_body_names


def _try_load_robot_spec_npz(robot_spec_npz: str, device: Union[str, torch.device]) -> Dict:
    """
    尽量复用你项目里的 load_robot_spec_npz；如果 import 不到，就用 np.load 兜底。
    """
    try:
        from worknet.robot_spec_encoder import load_robot_spec_npz as _loader  # type: ignore
        return _loader(robot_spec_npz, device=str(device))
    except Exception:
        npz = np.load(robot_spec_npz, allow_pickle=True)
        spec: Dict[str, object] = {}
        for k in npz.files:
            spec[k] = npz[k]
        return spec


def _get_first_existing(spec: Dict, keys: Sequence[str]):
    for k in keys:
        if k in spec:
            return spec[k]
    return None


def _as_torch(x, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)


def _as_long(x, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.long)
    return torch.as_tensor(np.asarray(x), device=device, dtype=torch.long)


def _topo_order(parent: torch.Tensor) -> List[int]:
    """
    parent: (N,) long, root parent = -1 或 自指
    生成拓扑序（父在前）。
    """
    N = int(parent.numel())
    children: List[List[int]] = [[] for _ in range(N)]
    roots = []
    for i in range(N):
        p = int(parent[i].item())
        if p < 0 or p == i:
            roots.append(i)
        else:
            children[p].append(i)

    order: List[int] = []
    stack = roots[:]
    while stack:
        u = stack.pop(0)
        order.append(u)
        stack.extend(children[u])
    if len(order) != N:
        seen = set(order)
        for i in range(N):
            if i not in seen:
                order.append(i)
    return order


# ============================================================
# RobotTorchFK (differentiable)
# ============================================================

class RobotTorchFK(nn.Module):
    """
    简化树形FK（足够支撑语义方向loss反传）：
      pos_i = pos_parent + R_parent * offset_i
      rot_i = rot_parent ⊗ rot_local(axis_i, angle_i)   (若该节点有 dof)

    重要：这里输出的是 robot_spec 的“关节 pivot/节点位置”。
    MuJoCo 的 data.xpos 是 body frame 原点，常与 pivot 不同；
    对比时建议用 data.xanchor（joint anchor）来对齐。
    """

    def __init__(self,
                 robot_spec: Dict,
                 device: Optional[Union[str, torch.device]] = None,
                 quat_order_root_in: str = "wxyz"):
        super().__init__()
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        parent = _get_first_existing(robot_spec, ["parent_idx", "parents", "parent"])
        if parent is None:
            raise ValueError("robot_spec missing parent_idx")
        self.parent_idx = _as_long(parent, self.device)

        # --- MOD: offset 优先使用 offset_vec（通常是 parent-frame 的关节pivot偏移）；否则退化到 offset_base/offset ---
        offset = _get_first_existing(robot_spec, ["offset_vec", "offset_vecs", "offset_base", "offset", "offsets", "joint_offset"])
        if offset is None:
            raise ValueError("robot_spec missing offset/offset_vec/offset_base")
        self.offset = _as_torch(offset, self.device)  # (N,3)

        axis = _get_first_existing(robot_spec, ["axis", "axes", "joint_axis", "joint_axes", "axis_local"])
        if axis is None:
            axis = np.zeros((self.offset.shape[0], 3), dtype=np.float32)
        self.axis = _as_torch(axis, self.device)  # (N,3)

        # --- MOD: dof_map 可选；若不存在，forward 里用 ndof==N 的启发式映射 did=i ---
        dof_map = _get_first_existing(robot_spec, ["dof_map", "dof_idx", "dof_id", "node_dof_idx"])
        self.dof_map = _as_long(dof_map, self.device) if dof_map is not None else None

        # names（用于从 ik_config 的 robot_body_name 找索引）
        names = _get_first_existing(robot_spec, ["body_names", "link_body_list", "node_names", "joint_names"])
        if names is not None:
            if isinstance(names, (list, tuple)):
                self.body_names = list(names)
            else:
                arr = np.asarray(names)
                if arr.dtype.type is np.str_ or arr.dtype == object:
                    self.body_names = [str(x) for x in arr.tolist()]
                else:
                    self.body_names = [str(x) for x in arr]
        else:
            self.body_names = None

        self.base_body_name = None
        try:
            bbn = robot_spec.get("base_body_name", None)
            if bbn is not None:
                self.base_body_name = str(bbn.item() if hasattr(bbn, "item") else bbn)
        except Exception:
            self.base_body_name = None

        self.order = _topo_order(self.parent_idx)
        self.quat_order_root_in = quat_order_root_in

    def get_body_index(self, name: str) -> int:
        if self.body_names is None:
            raise ValueError("robot_spec does not contain body/joint names; cannot map by name.")
        try:
            return self.body_names.index(name)
        except ValueError as e:
            raise ValueError(f"Robot body '{name}' not found in robot_spec names.") from e

    def forward(self,
                root_pos: torch.Tensor,
                root_rot: torch.Tensor,
                dof_pos: torch.Tensor,
                quat_order: Optional[str] = None,
                body_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        root_pos: (B,T,3)
        root_rot: (B,T,4) (wxyz by default)
        dof_pos : (B,T,ndof)
        body_indices: (J,) long -> 只输出这些 body 的 pos
        """
        if root_pos.dim() != 3:
            raise ValueError("root_pos must be (B,T,3)")
        if root_rot.dim() != 3:
            raise ValueError("root_rot must be (B,T,4)")
        if dof_pos.dim() != 3:
            raise ValueError("dof_pos must be (B,T,ndof)")

        B, T, _ = root_pos.shape
        N = int(self.parent_idx.numel())

        quat_order = quat_order or self.quat_order_root_in
        root_rot_wxyz = ensure_wxyz(root_rot, quat_order=quat_order)
        root_rot_wxyz = quat_normalize(root_rot_wxyz)

        pos = torch.zeros((B, T, N, 3), device=root_pos.device, dtype=root_pos.dtype)
        rot = torch.zeros((B, T, N, 4), device=root_pos.device, dtype=root_pos.dtype)

        # --- MOD: treat root_pos/root_rot as a *virtual base*; nodes with parent=-1 attach to this base
        base_pos = root_pos
        base_rot = root_rot_wxyz

        eye = torch.tensor([1.0, 0.0, 0.0, 0.0], device=root_pos.device, dtype=root_pos.dtype).view(1, 1, 4)
        ndof = int(dof_pos.shape[-1])

        for i in self.order:
            p = int(self.parent_idx[i].item())
            if p < 0 or p == i:
                parent_pos = base_pos
                parent_rot = base_rot
            else:
                parent_pos = pos[:, :, p, :]
                parent_rot = rot[:, :, p, :]

            off_i = self.offset[i].view(1, 1, 3).to(root_pos.device, root_pos.dtype)
            pos[:, :, i, :] = parent_pos + quat_apply(parent_rot, off_i)

            # --- MOD: dof indexing (prefer spec-provided dof_map; else heuristic)
            if self.dof_map is not None:
                did = int(self.dof_map[i].item())
            else:
                did = i if ndof == N else (i - 1 if ndof == N - 1 else -1)

            if 0 <= did < ndof:
                axis_i = self.axis[i].view(1, 1, 3).to(root_pos.device, root_pos.dtype)
                ang = dof_pos[:, :, did]
                local_q = quat_from_axis_angle(axis_i.expand(B, T, 3), ang)
            else:
                local_q = eye.expand(B, T, 4)

            rot[:, :, i, :] = quat_mul(parent_rot, local_q)

        if body_indices is not None:
            return pos[:, :, body_indices, :]
        return pos


# ============================================================
# Anchor map
# ============================================================

@dataclass
class AnchorMap:
    robot_names: List[str]
    human_names: List[str]
    pos_weights: torch.Tensor       # (J,)
    robot_indices: torch.Tensor     # (J,) 允许 -1 表示“用 root_pos”
    human_indices: List[int]
    root_index: int                # anchor list 中 root 的索引


def _infer_root_index(robot_names: List[str], root_robot: Optional[str]) -> int:
    if root_robot is None:
        return 0
    if root_robot in robot_names:
        return robot_names.index(root_robot)
    return 0


def build_anchor_map(robot_fk: RobotTorchFK,
                     human_body_names: Sequence[str],
                     ik_cfg: dict,
                     tables: str,
                     root_robot: Optional[str],
                     device: torch.device) -> AnchorMap:
    """
    - robot_indices: 来自 robot_fk.get_body_index
    - 但如果 r_name == root_robot 或 r_name == robot_fk.base_body_name 且不在 body_names，
      则用 -1 作为索引（表示 anchor 点直接使用 root_pos）。
    """
    pairs = build_pairs_from_ik_config(ik_cfg, tables=tables)

    human_body_names = list(human_body_names)

    robot_names: List[str] = []
    human_names: List[str] = []
    pos_w: List[float] = []

    robot_indices: List[int] = []
    human_indices: List[int] = []

    base_name = robot_fk.base_body_name

    for r_name, h_name, w in pairs:
        if h_name not in human_body_names:
            raise ValueError(f"Human joint '{h_name}' not found in link_body_list / human_body_names.")
        human_idx = human_body_names.index(h_name)

        # --- MOD: root 名称允许不出现在 body_names（用 -1 表示用 root_pos）---
        if (root_robot is not None and r_name == root_robot) or (base_name is not None and r_name == base_name):
            try:
                r_idx = robot_fk.get_body_index(r_name)
            except Exception:
                r_idx = -1
        else:
            r_idx = robot_fk.get_body_index(r_name)

        robot_names.append(r_name)
        human_names.append(h_name)
        pos_w.append(float(w))
        robot_indices.append(int(r_idx))
        human_indices.append(int(human_idx))

    root_index = _infer_root_index(robot_names, root_robot or base_name)

    return AnchorMap(
        robot_names=robot_names,
        human_names=human_names,
        pos_weights=torch.as_tensor(pos_w, device=device, dtype=torch.float32),
        robot_indices=torch.as_tensor(robot_indices, device=device, dtype=torch.long),
        human_indices=human_indices,
        root_index=root_index,
    )


# ============================================================
# SemanticLossComputer (for training)
# ============================================================

class SemanticLossComputer(nn.Module):
    """
    训练时用法（推荐）：
      sem = SemanticLossComputer(robot_spec_npz, ik_config, tables, device, root_robot="pelvis",
                                human_body_names=ds_train.human_body_names_or_link_list)
      loss_sem = sem(pred_dof, pred_root, xj=xj)   # 用 xj 抽 human anchors

    注意：这里默认 xj[..., :3] 是 human joint positions，顺序与 human_body_names 一致。
    """

    def __init__(self,
                 robot_spec_npz: str,
                 ik_config: str,
                 tables: str = "both",
                 device: Union[str, torch.device] = "cpu",
                 root_robot: Optional[str] = None,
                 pred_root_quat_order: str = "xyzw",
                 human_body_names: Optional[Sequence[str]] = None,
                 human_pos_dim_slice: Tuple[int, int] = (0, 3)):
        super().__init__()
        self.device = torch.device(device)
        self.tables = tables
        self.pred_root_quat_order = pred_root_quat_order
        self.human_pos_slice = human_pos_dim_slice

        self.robot_spec = _try_load_robot_spec_npz(robot_spec_npz, device=self.device)
        self.fk = RobotTorchFK(self.robot_spec, device=self.device, quat_order_root_in="wxyz")

        with open(ik_config, "r", encoding="utf-8") as f:
            self.ik_cfg = json.load(f)

        self.human_body_names = list(human_body_names) if human_body_names is not None else None

        self.root_robot = root_robot
        self._anchor_map: Optional[AnchorMap] = None

    def _get_anchor_map(self, human_body_names: Sequence[str]) -> AnchorMap:
        if self._anchor_map is None:
            self._anchor_map = build_anchor_map(
                robot_fk=self.fk,
                human_body_names=human_body_names,
                ik_cfg=self.ik_cfg,
                tables=self.tables,
                root_robot=self.root_robot,
                device=self.device,
            )
        return self._anchor_map

    @staticmethod
    def _gather_human_from_xj(xj: torch.Tensor,
                             human_body_names: Sequence[str],
                             human_anchor_names: Sequence[str],
                             pos_slice: Tuple[int, int]) -> torch.Tensor:
        """
        xj: (B,T,Nh,Dj)
        return: (B,T,J,3) 取 xj[..., pos_slice[0]:pos_slice[1]] 作为位置
        """
        B, T, Nh, Dj = xj.shape
        s0, s1 = pos_slice
        if not (0 <= s0 < s1 <= Dj):
            raise ValueError(f"Invalid human_pos_dim_slice={pos_slice} for Dj={Dj}")
        pos = xj[..., s0:s1]  # (B,T,Nh,3)

        name_to_idx = {str(n): i for i, n in enumerate(human_body_names)}
        idx = []
        for n in human_anchor_names:
            if n not in name_to_idx:
                raise ValueError(f"Human joint '{n}' not found in provided human_body_names.")
            idx.append(name_to_idx[n])
        idx_t = torch.as_tensor(idx, device=pos.device, dtype=torch.long)
        return pos.index_select(dim=2, index=idx_t)  # (B,T,J,3)

    def _robot_anchors_from_pred(self,
                                pred_dof: torch.Tensor,
                                pred_root: torch.Tensor,
                                anchor_map: AnchorMap) -> torch.Tensor:
        """
        pred_dof: (B,T,Ndof)
        pred_root: (B,T,7 or >=7) [root_pos(3), root_quat(4,...)]
        return: (B,T,J,3)
        """
        root_pos = pred_root[..., :3]
        root_quat_in = pred_root[..., 3:7]
        root_quat_wxyz = ensure_wxyz(root_quat_in, quat_order=self.pred_root_quat_order)

        idx = anchor_map.robot_indices.to(pred_dof.device)
        idx_clamped = idx.clamp_min(0)

        x_r = self.fk(root_pos, root_quat_wxyz, pred_dof, quat_order="wxyz", body_indices=idx_clamped)  # (B,T,J,3)

        mask_root = (idx < 0)
        if mask_root.any():
            x_r = x_r.clone()
            k = int(mask_root.sum().item())
            x_r[:, :, mask_root, :] = root_pos.unsqueeze(2).expand(-1, -1, k, -1)
        return x_r

    def forward(self,
                pred_dof: torch.Tensor,
                pred_root: torch.Tensor,
                *,
                xj: Optional[torch.Tensor] = None,
                human_anchor_pos: Optional[torch.Tensor] = None,
                human_body_names: Optional[Sequence[str]] = None,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算语义方向 loss（mean(1-cos)）。

        输入任选其一：
        - human_anchor_pos: (B,T,J,3)  (最精确，直接给 anchors)
        - xj + human_body_names: 从 human joints features 中抽取 anchors
        - xj + (初始化时提供 human_body_names)

        返回：
          loss_sem (scalar)
        """
        if human_anchor_pos is None:
            if xj is None:
                raise ValueError("Need either human_anchor_pos or xj to compute semantic loss.")
            hb = list(human_body_names) if human_body_names is not None else self.human_body_names
            if hb is None:
                raise ValueError("human_body_names is required (pass to ctor or forward).")
            anchor_map = self._get_anchor_map(hb)
            x_h = self._gather_human_from_xj(xj, hb, anchor_map.human_names, self.human_pos_slice)
        else:
            if human_anchor_pos.dim() != 4:
                raise ValueError("human_anchor_pos must be (B,T,J,3)")
            hb = list(human_body_names) if human_body_names is not None else self.human_body_names
            if hb is None:
                raise ValueError("human_body_names required to align anchors ordering with ik_config.")
            anchor_map = self._get_anchor_map(hb)
            x_h = human_anchor_pos

        anchor_map = self._get_anchor_map(hb)
        x_r = self._robot_anchors_from_pred(pred_dof, pred_root, anchor_map)

        w = weights if weights is not None else anchor_map.pos_weights
        return semantic_direction_loss(x_h, x_r, root_index=anchor_map.root_index, weights=w)


# ============================================================
# CLI: compare torchFK vs mujocoFK
# ============================================================

def _build_robot_qpos(robot_motion: dict) -> np.ndarray:
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
    nq = 3 + 4 + dof_pos.shape[1]
    qpos = np.zeros((T, nq), dtype=float)

    qpos[:, :3] = root_pos
    qpos[:, 3:7] = root_rot_xyzw[:, [3, 0, 1, 2]]  # xyzw -> wxyz
    qpos[:, 7:] = dof_pos
    return qpos


def _mujoco_fk_points(robot_xml: str,
                      qpos_seq: np.ndarray,
                      robot_body_names: Sequence[str],
                      point_mode: str = "joint_xanchor") -> np.ndarray:
    """
    返回 (T,J,3) 的 MuJoCo 世界坐标。
    point_mode:
      - "body_xpos"     : data.xpos[body]（body frame 原点）
      - "joint_xanchor" : 使用 child body 的第一个 joint 的 data.xanchor[jid]（更接近关节 pivot）
    """
    try:
        import mujoco as mj
    except Exception as e:
        raise RuntimeError("MuJoCo not available. Install mujoco or run without --robot_xml.") from e

    model = mj.MjModel.from_xml_path(robot_xml)
    data = mj.MjData(model)

    body_ids = []
    for name in robot_body_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Body '{name}' not found in model.")
        body_ids.append(bid)

    joint_ids = []
    if point_mode == "joint_xanchor":
        for bid in body_ids:
            adr = int(model.body_jntadr[bid])
            num = int(model.body_jntnum[bid])
            if num <= 0:
                joint_ids.append(-1)
            else:
                joint_ids.append(adr)  # 第一个 joint
    else:
        joint_ids = [-1] * len(body_ids)

    T = qpos_seq.shape[0]
    J = len(body_ids)
    out = np.zeros((T, J, 3), dtype=float)

    for t in range(T):
        data.qpos[:] = qpos_seq[t]
        mj.mj_forward(model, data)
        for j, bid in enumerate(body_ids):
            if point_mode == "joint_xanchor":
                jid = joint_ids[j]
                if jid >= 0:
                    out[t, j] = data.xanchor[jid]
                else:
                    out[t, j] = data.xpos[bid]
            else:
                out[t, j] = data.xpos[bid]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=str, required=True)
    ap.add_argument("--robot_spec_npz", type=str, required=True)
    ap.add_argument("--ik_config", type=str, required=True)
    ap.add_argument("--tables", type=str, default="both", choices=["1", "2", "both"])
    ap.add_argument("--root_robot", type=str, default=None, help="例如 pelvis；默认用 robot_spec.base_body_name")
    ap.add_argument("--robot_xml", type=str, default=None, help="可选：提供则用 MuJoCo 做 FK 对比验证")
    ap.add_argument("--mujoco_point", type=str, default="joint_xanchor", choices=["joint_xanchor", "body_xpos"])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_frames", type=int, default=0, help=">0 则只取前N帧做验证加速")
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)
    if "robot_motion" not in payload or "human_motion" not in payload:
        raise ValueError("pkl 顶层必须包含 robot_motion 和 human_motion")

    robot_motion = payload["robot_motion"]
    human_motion = payload["human_motion"]

    device = torch.device(args.device)

    robot_spec = _try_load_robot_spec_npz(args.robot_spec_npz, device=device)
    fk = RobotTorchFK(robot_spec, device=device, quat_order_root_in="wxyz")
    base_body_name = fk.base_body_name

    with open(args.ik_config, "r", encoding="utf-8") as f:
        ik_cfg = json.load(f)

    human_body_names = list(human_motion["link_body_list"])
    root_robot = args.root_robot or base_body_name
    anchor_map = build_anchor_map(
        robot_fk=fk,
        human_body_names=human_body_names,
        ik_cfg=ik_cfg,
        tables=args.tables,
        root_robot=root_robot,
        device=device,
    )

    human_anchor_pos_np, _, _ = human_global_anchors_from_motion(human_motion, anchor_map.human_names)

    T_r = np.asarray(robot_motion["root_pos"]).shape[0]
    T_h = human_anchor_pos_np.shape[0]
    T = min(T_r, T_h)
    if args.max_frames and args.max_frames > 0:
        T = min(T, args.max_frames)
    human_anchor_pos_np = human_anchor_pos_np[:T]

    root_pos = torch.as_tensor(np.asarray(robot_motion["root_pos"], dtype=np.float32), device=device).unsqueeze(0)[:, :T]
    root_rot_xyzw = torch.as_tensor(np.asarray(robot_motion["root_rot"], dtype=np.float32), device=device).unsqueeze(0)[:, :T]
    root_rot_wxyz = ensure_wxyz(root_rot_xyzw, quat_order="xyzw")
    dof_pos = torch.as_tensor(np.asarray(robot_motion["dof_pos"], dtype=np.float32), device=device).unsqueeze(0)[:, :T]

    with torch.no_grad():
        idx = anchor_map.robot_indices.to(device)
        idx_clamped = idx.clamp_min(0)
        x_r = fk(root_pos, root_rot_wxyz, dof_pos, quat_order="wxyz", body_indices=idx_clamped)  # (1,T,J,3)
        mask_root = (idx < 0)
        if mask_root.any():
            x_r = x_r.clone()
            k = int(mask_root.sum().item())
            x_r[:, :, mask_root, :] = root_pos.unsqueeze(2).expand(-1, -1, k, -1)

    x_h_t = torch.as_tensor(human_anchor_pos_np, device=device, dtype=torch.float32).unsqueeze(0)  # (1,T,J,3)
    s_err = semantic_direction_loss(x_h_t, x_r.to(torch.float32), root_index=anchor_map.root_index, weights=None).item()
    print(f"[semantic] root_index={anchor_map.root_index} root_name={anchor_map.robot_names[anchor_map.root_index]}")
    print(f"  direction_loss(mean(1-cos)) = {s_err:.6f}")

    if args.robot_xml:
        qpos = _build_robot_qpos(robot_motion)[:T]
        x_mj = _mujoco_fk_points(args.robot_xml, qpos, anchor_map.robot_names, point_mode=args.mujoco_point)  # (T,J,3)

        x_r_np = x_r.squeeze(0).cpu().numpy()
        diff = x_r_np - x_mj
        l2 = np.linalg.norm(diff, axis=-1)  # (T,J)
        print(f"[compare] torchFK(pivot) vs mujoco({args.mujoco_point}): mean={l2.mean():.6f}  max={l2.max():.6f}")
        print("  note: 若用 body_xpos，误差大通常是“body 原点 != 关节 pivot”，不代表 torchFK 错。")
    else:
        print("[compare] robot_xml not provided, skip mujoco comparison.")

    print("[anchors] first 12 pairs:")
    for i in range(min(12, len(anchor_map.robot_names))):
        print(f"  {i:02d}: robot={anchor_map.robot_names[i]} | human={anchor_map.human_names[i]}")


if __name__ == "__main__":
    main()
