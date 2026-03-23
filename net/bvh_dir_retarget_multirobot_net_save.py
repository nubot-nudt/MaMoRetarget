# -*- coding: utf-8 -*-
"""
Multi-robot batch retarget:
  BVH -> (per-robot GMR scaled_human_data) -> multi-robot net -> save pkl

Output format (EXACTLY same as bvh_dir_retarget_net_save.py):
save_data = {
  "robot_motion": {
      "fps": 30,
      "root_pos": (T,3),
      "root_rot": (T,4) xyzw,
      "dof_pos":  (T,DoF),
      "dof_names": list[str] or None,
      "local_body_pos": None,
      "link_body_list": None,
  },
  "human_motion": {
      "fps": 30,
      "root_pos": (T,3),
      "root_rot": (T,4) xyzw,
      "global_body_pos": (T,Nh,3),
      "global_body_rot": (T,Nh,4) xyzw,
      "local_body_pos": (T,Nh,3),
      "local_body_rot": (T,Nh,4) xyzw,
      "link_body_list": list[str],
      "note": "normal GMR scaled_human_data (uses human_scale_table)",
  }
}
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.lafan1 import load_lafan1_file

# reuse your training dataset feature builders
from dataset.retarget_io_inout_dataset import (
    build_X_joint_from_human_motion,
    build_X_root_from_human_motion,
    load_morph_npz,
)

# reuse your trained model modules
from worknet.human_encoder import HumanEncoder
from worknet.robot_spec_encoder import load_robot_spec_npz, build_robot_node_features, RobotSpecEncoder
from worknet.cross_attn_decoder import CrossAttnRetargetDecoder


# ---------------- Quaternion helpers (wxyz) ----------------
def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    out = q.copy()
    out[..., 1:] *= -1.0
    return out

def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, both wxyz"""
    aw, ax, ay, az = np.split(a, 4, axis=-1)
    bw, bx, by, bz = np.split(b, 4, axis=-1)
    w = aw*bw - ax*bx - ay*by - az*bz
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    return np.concatenate([w, x, y, z], axis=-1)

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


def build_human_target_list(retargeter: GMR) -> List[str]:
    """
    从 ik_match_table1/2 中提取 human target joints 的稳定顺序（按 table 遍历顺序，去重）。
    只取 pos_weight!=0 或 rot_weight!=0 的项（真正作为 target 的）。
    """
    names: List[str] = []

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

    root_name = retargeter.human_root_name
    if root_name not in names:
        names = [root_name] + names
    return names


def extract_targets_from_scaled_human_data(scaled_human_data: dict, body_names: list) -> Tuple[np.ndarray, np.ndarray]:
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
            raise KeyError(
                f"scaled_human_data missing key '{n}'. "
                f"Available keys (head): {list(scaled_human_data.keys())[:10]} ..."
            )
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

    root_wxyz = xyzw_to_wxyz(root_rot)               # (T,4)
    joint_wxyz = xyzw_to_wxyz(rot_global_xyzw)       # (T,N,4)
    root_inv = quat_inv_wxyz(root_wxyz)[:, None, :]  # (T,1,4)
    local_wxyz = quat_mul_wxyz(root_inv, joint_wxyz)
    local_rot_xyzw = wxyz_to_xyzw(local_wxyz).astype(np.float32)

    return (
        root_pos.astype(np.float32),
        root_rot.astype(np.float32),
        local_pos.astype(np.float32),
        local_rot_xyzw,
    )


# -------- rot6d -> quat(xyzw) (torch) --------
def rot6d_to_rotmat(rot6d: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    rot6d: (...,6) -> R: (...,3,3) with columns [b1,b2,b3]
    """
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=eps)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    a2_ortho = a2 - dot * b1
    b2 = F.normalize(a2_ortho, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)

def rotmat_to_quat_xyzw(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    R: (...,3,3) -> quat (...,4) xyzw
    """
    m00 = R[..., 0, 0]
    m11 = R[..., 1, 1]
    m22 = R[..., 2, 2]
    trace = m00 + m11 + m22

    def _safe_sqrt(x):
        return torch.sqrt(torch.clamp(x, min=0.0) + eps)

    qw = torch.zeros_like(trace)
    qx = torch.zeros_like(trace)
    qy = torch.zeros_like(trace)
    qz = torch.zeros_like(trace)

    mask0 = trace > 0.0
    s0 = _safe_sqrt(trace[mask0] + 1.0) * 2.0
    qw[mask0] = 0.25 * s0
    qx[mask0] = (R[..., 2, 1][mask0] - R[..., 1, 2][mask0]) / s0
    qy[mask0] = (R[..., 0, 2][mask0] - R[..., 2, 0][mask0]) / s0
    qz[mask0] = (R[..., 1, 0][mask0] - R[..., 0, 1][mask0]) / s0

    mask1 = (~mask0) & (m00 > m11) & (m00 > m22)
    s1 = _safe_sqrt(1.0 + m00[mask1] - m11[mask1] - m22[mask1]) * 2.0
    qw[mask1] = (R[..., 2, 1][mask1] - R[..., 1, 2][mask1]) / s1
    qx[mask1] = 0.25 * s1
    qy[mask1] = (R[..., 0, 1][mask1] + R[..., 1, 0][mask1]) / s1
    qz[mask1] = (R[..., 0, 2][mask1] + R[..., 2, 0][mask1]) / s1

    mask2 = (~mask0) & (~mask1) & (m11 > m22)
    s2 = _safe_sqrt(1.0 + m11[mask2] - m00[mask2] - m22[mask2]) * 2.0
    qw[mask2] = (R[..., 0, 2][mask2] - R[..., 2, 0][mask2]) / s2
    qx[mask2] = (R[..., 0, 1][mask2] + R[..., 1, 0][mask2]) / s2
    qy[mask2] = 0.25 * s2
    qz[mask2] = (R[..., 1, 2][mask2] + R[..., 2, 1][mask2]) / s2

    mask3 = (~mask0) & (~mask1) & (~mask2)
    s3 = _safe_sqrt(1.0 + m22[mask3] - m00[mask3] - m11[mask3]) * 2.0
    qw[mask3] = (R[..., 1, 0][mask3] - R[..., 0, 1][mask3]) / s3
    qx[mask3] = (R[..., 0, 2][mask3] + R[..., 2, 0][mask3]) / s3
    qy[mask3] = (R[..., 1, 2][mask3] + R[..., 2, 1][mask3]) / s3
    qz[mask3] = 0.25 * s3

    q = torch.stack([qx, qy, qz, qw], dim=-1)
    q = F.normalize(q, dim=-1, eps=1e-8)
    return q

def rot6d_to_quat_xyzw(rot6d: torch.Tensor) -> torch.Tensor:
    R = rot6d_to_rotmat(rot6d)
    return rotmat_to_quat_xyzw(R)

def enforce_quat_continuity_xyzw(q: np.ndarray) -> np.ndarray:
    """flip sign to keep consecutive dot>=0 (avoid visual jitter)"""
    q = q.copy()
    for t in range(1, q.shape[0]):
        if np.dot(q[t-1], q[t]) < 0:
            q[t] *= -1.0
    return q


def extract_dof_names(robot_spec: dict, dof_dim: int) -> Optional[List[str]]:
    """best-effort: try common keys"""
    for k in ["dof_names", "actuated_dof_names", "joint_names", "actuated_joint_names"]:
        if k in robot_spec:
            names = robot_spec[k]
            if isinstance(names, (list, tuple)) and len(names) >= dof_dim:
                return list(names)[:dof_dim]
    return None


@torch.no_grad()
def infer_full_sequence(
    human_motion: dict,
    mh_vec: np.ndarray,
    human_encoder: HumanEncoder,
    robot_encoder: RobotSpecEncoder,
    decoder: CrossAttnRetargetDecoder,
    robot_spec: dict,
    node_feat: torch.Tensor,
    device: torch.device,
    clip_len: int = 64,
    stride: int = 16,
    use_vel_joint: bool = True,
    use_vel_root: bool = True,
    vel_in_mps: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return:
      root_pos (T,3), root_rot_xyzw (T,4), dof_pos (T,DoF)
    """
    x_joint, _ = build_X_joint_from_human_motion(
        human_motion, use_vel=use_vel_joint, fps_default=30.0, vel_in_mps=vel_in_mps
    )
    x_root = build_X_root_from_human_motion(
        human_motion, use_vel=use_vel_root, fps_default=30.0, vel_in_mps=vel_in_mps
    )
    T = x_joint.shape[0]

    xj = torch.from_numpy(x_joint).to(device=device, dtype=torch.float32)   # (T,Nh,Dj)
    xr = torch.from_numpy(x_root).to(device=device, dtype=torch.float32)    # (T,Dr)
    mh = torch.from_numpy(mh_vec.astype(np.float32)).to(device=device, dtype=torch.float32)  # (M,)

    # precompute robot embedding (depends only on robot)
    R = robot_encoder(node_feat, robot_spec["parent_idx"], robot_spec["joint_type"])

    y_dof_acc = None
    y_root_acc = None
    w_acc = torch.zeros((T, 1), device=device, dtype=torch.float32)

    def _get_clip(t0: int):
        t1 = min(t0 + clip_len, T)
        valid = t1 - t0
        xj_clip = xj[t0:t1]
        xr_clip = xr[t0:t1]
        if valid < clip_len:
            pad_n = clip_len - valid
            xj_pad = xj_clip[-1:].repeat(pad_n, 1, 1)
            xr_pad = xr_clip[-1:].repeat(pad_n, 1)
            xj_clip = torch.cat([xj_clip, xj_pad], dim=0)
            xr_clip = torch.cat([xr_clip, xr_pad], dim=0)
        return xj_clip, xr_clip, valid

    for t0 in range(0, T, stride):
        xj_clip, xr_clip, valid = _get_clip(t0)
        xj_b = xj_clip.unsqueeze(0)  # (1,L,Nh,Dj)
        xr_b = xr_clip.unsqueeze(0)  # (1,L,Dr)
        mh_b = mh.unsqueeze(0)       # (1,M)

        H_all, _, _ = human_encoder(xj_b, xr_b, mh_b)
        y_dof_pred, y_root_pred, _ = decoder(H_all, R, robot_spec, return_attn=False)
        y_dof_pred = y_dof_pred[0, :valid]
        y_root_pred = y_root_pred[0, :valid]

        if y_dof_acc is None:
            dof_dim = y_dof_pred.shape[-1]
            y_dof_acc = torch.zeros((T, dof_dim), device=device, dtype=torch.float32)
            y_root_acc = torch.zeros((T, y_root_pred.shape[-1]), device=device, dtype=torch.float32)

        y_dof_acc[t0:t0+valid] += y_dof_pred
        y_root_acc[t0:t0+valid] += y_root_pred
        w_acc[t0:t0+valid] += 1.0

    y_dof = y_dof_acc / torch.clamp(w_acc, min=1.0)
    y_root = y_root_acc / torch.clamp(w_acc, min=1.0)

    root_pos = y_root[:, 0:3]
    root_rot6d = y_root[:, 3:9]
    root_rot_xyzw = rot6d_to_quat_xyzw(root_rot6d)

    root_pos_np = root_pos.detach().cpu().numpy().astype(np.float32)
    root_rot_np = root_rot_xyzw.detach().cpu().numpy().astype(np.float32)
    root_rot_np = enforce_quat_continuity_xyzw(root_rot_np)

    dof_np = y_dof.detach().cpu().numpy().astype(np.float32)
    return root_pos_np, root_rot_np, dof_np


def list_bvh_files(bvh_dir: str, recursive: bool = True) -> List[Path]:
    base = Path(bvh_dir)
    if recursive:
        return sorted(base.rglob("*.bvh"))
    return sorted(base.glob("*.bvh"))


def load_morph_for_motion(morph_npz: str, morph_map: Optional[Dict[str, str]], motion_id: str) -> np.ndarray:
    if morph_map is None:
        return load_morph_npz(morph_npz)
    if motion_id not in morph_map:
        raise KeyError(f"morph_map missing key '{motion_id}' (stem of BVH filename).")
    return load_morph_npz(morph_map[motion_id])


def load_robots_json(path: str) -> List[Dict]:
    if not path or not os.path.isfile(path):
        raise RuntimeError(f"robots_json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if isinstance(cfg, dict):
        robots = cfg.get("robots", None)
        if robots is None:
            raise RuntimeError("robots_json is a dict but missing key 'robots'")
    elif isinstance(cfg, list):
        robots = cfg
    else:
        raise RuntimeError("robots_json must be a list or a dict with key 'robots'")
    out = []
    for r in robots:
        if not isinstance(r, dict):
            raise RuntimeError("Each robot entry must be a dict")
        if "name" not in r:
            raise RuntimeError("Each robot entry must have 'name'")
        if "robot_spec_npz" not in r:
            raise RuntimeError(f"Robot {r.get('name')} missing robot_spec_npz")
        if "morph_npz" not in r:
            raise RuntimeError(f"Robot {r.get('name')} missing morph_npz")
        out.append(r)
    return out


def infer_gmr_robot_name(robot_entry: Dict) -> str:
    # allow explicit override
    for k in ["gmr_robot", "tgt_robot", "robot"]:
        if k in robot_entry and robot_entry[k]:
            return str(robot_entry[k])

    # fallback mapping by "name"
    name = str(robot_entry.get("name", "")).lower()
    fallback = {
        "g1": "unitree_g1",
        "g1_with_hands": "unitree_g1_with_hands",
        "t1": "booster_t1",
        "toddy": "stanford_toddy",
        "n1": "fourier_n1",
        "pm01": "engineai_pm01",
        "zlite": "zlite",
    }
    if name in fallback:
        return fallback[name]

    # last resort: assume name itself is valid for GMR
    return str(robot_entry.get("name"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh_dir", required=True, type=str, help="folder containing .bvh files")
    ap.add_argument("--out_dir", required=True, type=str, help="output root folder; saves under out_dir/<robot_name>/*.pkl")
    ap.add_argument("--ckpt", required=True, type=str, help="path to best.pt/last.pt (multi-robot)")
    ap.add_argument("--robots_json", required=True, type=str, help="json list/dict of robots (robot_spec_npz, morph_npz, ...)")
    ap.add_argument("--morph_map_json", type=str, default=None,
                    help="optional: json dict {motion_id: morph_npz_path} (applied to ALL robots unless robot entry overrides)")
    ap.add_argument("--recursive", action="store_true", default=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--offset_to_ground", action="store_true", default=True)

    # must match training hyperparams
    ap.add_argument("--clip_len", type=int, default=64)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--human_layers", type=int, default=4)
    ap.add_argument("--robot_layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no_vel", action="store_true")
    ap.add_argument("--vel_in_mps", action="store_true")
    ap.add_argument("--no_morph_film", action="store_true")

    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--amp", action="store_true")

    args = ap.parse_args()
    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # optional global morph map
    morph_map_global = None
    if args.morph_map_json is not None:
        with open(args.morph_map_json, "r", encoding="utf-8") as f:
            morph_map_global = json.load(f)
        if not isinstance(morph_map_global, dict):
            raise TypeError("morph_map_json must be a dict {motion_id: npz_path}")

    robots = load_robots_json(args.robots_json)

    # pre-load robot specs + node features
    robot_specs: Dict[str, dict] = {}
    robot_node_feat: Dict[str, torch.Tensor] = {}
    robot_morph_npz: Dict[str, str] = {}
    robot_gmr_name: Dict[str, str] = {}
    robot_morph_map: Dict[str, Optional[Dict[str, str]]] = {}

    for r in robots:
        name = str(r["name"])
        spec_npz = str(r["robot_spec_npz"])
        morph_npz = str(r["morph_npz"])
        gmr_name = infer_gmr_robot_name(r)

        # per-robot morph map override (optional)
        mm = morph_map_global
        if "morph_map_json" in r and r["morph_map_json"]:
            with open(str(r["morph_map_json"]), "r", encoding="utf-8") as f:
                mm = json.load(f)
            if not isinstance(mm, dict):
                raise TypeError(f"Robot {name}: morph_map_json must be dict {{motion_id: npz_path}}")

        spec = load_robot_spec_npz(spec_npz, device=str(device))
        node_feat = build_robot_node_features(spec, use_offset_vec=True, use_limits=True).to(device)

        robot_specs[name] = spec
        robot_node_feat[name] = node_feat
        robot_morph_npz[name] = morph_npz
        robot_gmr_name[name] = gmr_name
        robot_morph_map[name] = mm

        print(f"[Robot {name}] gmr_robot={gmr_name} | spec={spec_npz} | morph={morph_npz}")

    ckpt = torch.load(args.ckpt, map_location=device)

    bvh_files = list_bvh_files(args.bvh_dir, recursive=args.recursive)
    if len(bvh_files) == 0:
        raise RuntimeError(f"No .bvh found under {args.bvh_dir}")

    human_encoder: Optional[HumanEncoder] = None
    robot_encoder: Optional[RobotSpecEncoder] = None
    decoder: Optional[CrossAttnRetargetDecoder] = None

    # sanity: multi-robot should share (Nh,Dj,Dr,morph_dim)
    shared_dims = None  # (Nh,Dj,Dr,morph_dim)
    shared_body_names = None  # enforce Nh consistency at source

    use_vel = (not args.no_vel)

    for bvh_path in tqdm(bvh_files, desc="BVH -> MultiRobotNet -> PKL"):
        rel = bvh_path.relative_to(Path(args.bvh_dir))
        motion_id = bvh_path.stem

        # load BVH once
        lafan1_frames, actual_human_height = load_lafan1_file(str(bvh_path))
        motion_fps = 30

        for robot_name in robot_specs.keys():
            out_path = out_root / robot_name / rel.with_suffix(".pkl")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and (not args.overwrite):
                continue

            # ---------------- build human_motion for THIS robot (normal GMR scaled_human_data) ----------------
            gmr_robot = robot_gmr_name[robot_name]
            retargeter = GMR(src_human="bvh", tgt_robot=gmr_robot, actual_human_height=actual_human_height)

            body_names = build_human_target_list(retargeter)
            root_name = retargeter.human_root_name
            if root_name not in body_names:
                body_names = [root_name] + body_names
            root_idx = body_names.index(root_name)

            # enforce shared Nh across robots (multi-robot network expects same human feature construction)
            if shared_body_names is None:
                shared_body_names = list(body_names)
            else:
                if list(body_names) != list(shared_body_names):
                    raise RuntimeError(
                        f"[ERR] Human target list mismatch across robots.\n"
                        f"  first_robot_body_names={shared_body_names}\n"
                        f"  robot={robot_name} body_names={body_names}\n"
                        f"Hint: multi-robot training assumes same Nh and same human targets ordering."
                    )

            hg_pos_list = []
            hg_rot_list = []
            for frame in lafan1_frames:
                if hasattr(retargeter, "update_targets"):
                    retargeter.update_targets(frame, offset_to_ground=args.offset_to_ground)
                else:
                    _ = retargeter.retarget(frame, offset_to_ground=args.offset_to_ground)

                posg, rotg = extract_targets_from_scaled_human_data(retargeter.scaled_human_data, body_names)
                hg_pos_list.append(posg)
                hg_rot_list.append(rotg)

            hg_pos = np.stack(hg_pos_list, axis=0).astype(np.float32)  # (T,Nh,3)
            hg_rot = np.stack(hg_rot_list, axis=0).astype(np.float32)  # (T,Nh,4) xyzw
            hg_root_pos, hg_root_rot, hg_local_pos, hg_local_rot = compute_root_relative(hg_pos, hg_rot, root_idx)

            human_motion = {
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

            # ---------------- dims & lazy-init networks ----------------
            xj, _ = build_X_joint_from_human_motion(
                human_motion, use_vel=use_vel, fps_default=30.0, vel_in_mps=args.vel_in_mps
            )
            xr = build_X_root_from_human_motion(
                human_motion, use_vel=use_vel, fps_default=30.0, vel_in_mps=args.vel_in_mps
            )
            Nh, Dj = xj.shape[1], xj.shape[2]
            Dr = xr.shape[1]

            mh_vec = load_morph_for_motion(robot_morph_npz[robot_name], robot_morph_map[robot_name], motion_id)
            morph_dim = int(mh_vec.reshape(-1).shape[0])

            dims = (Nh, Dj, Dr, morph_dim)
            if shared_dims is None:
                shared_dims = dims
            else:
                if dims != shared_dims:
                    raise RuntimeError(
                        f"[ERR] Shared dims mismatch across robots/files.\n"
                        f"  expected={shared_dims} but got={dims} for robot={robot_name}, bvh={bvh_path}\n"
                        f"Hint: multi-robot training assumes SAME Nh/Dj/Dr/morph_dim."
                    )

            if human_encoder is None:
                human_encoder = HumanEncoder(
                    Nh=Nh, Dj=Dj, Dr=Dr,
                    d_model=args.d_model,
                    n_heads=args.heads,
                    n_layers=args.human_layers,
                    dropout=args.dropout,
                    max_len=max(1024, args.clip_len),
                    use_morph_film=(not args.no_morph_film),
                    morph_dim=morph_dim,
                ).to(device)

                # NOTE: node_feat_dim must match all robots (build_robot_node_features should be consistent)
                any_robot = next(iter(robot_node_feat.values()))
                robot_encoder = RobotSpecEncoder(
                    node_feat_dim=any_robot.shape[-1],
                    d_model=args.d_model,
                    num_layers=args.robot_layers,
                    dropout=args.dropout,
                ).to(device)

                decoder = CrossAttnRetargetDecoder(
                    d_model=args.d_model,
                    n_heads=args.heads,
                    dropout=args.dropout,
                    root_out_dim=9,
                    use_root_head=True,
                    limit_mode="tanh",
                ).to(device)

                # load weights
                def _load_sd(mod, key):
                    if key not in ckpt:
                        raise KeyError(f"ckpt missing '{key}'")
                    mod.load_state_dict(ckpt[key], strict=True)

                _load_sd(human_encoder, "human_encoder")
                _load_sd(robot_encoder, "robot_encoder")
                _load_sd(decoder, "decoder")

                human_encoder.eval()
                robot_encoder.eval()
                decoder.eval()

            assert human_encoder is not None and robot_encoder is not None and decoder is not None

            # ---------------- inference ----------------
            autocast_ctx = torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda"))
            with autocast_ctx:
                root_pos, root_rot_xyzw, dof_pos = infer_full_sequence(
                    human_motion=human_motion,
                    mh_vec=mh_vec,
                    human_encoder=human_encoder,
                    robot_encoder=robot_encoder,
                    decoder=decoder,
                    robot_spec=robot_specs[robot_name],
                    node_feat=robot_node_feat[robot_name],
                    device=device,
                    clip_len=args.clip_len,
                    stride=args.stride,
                    use_vel_joint=use_vel,
                    use_vel_root=use_vel,
                    vel_in_mps=args.vel_in_mps,
                )

            dof_names = extract_dof_names(robot_specs[robot_name], dof_pos.shape[1])

            robot_motion = {
                "fps": motion_fps,
                "root_pos": root_pos,
                "root_rot": root_rot_xyzw,   # xyzw
                "dof_pos": dof_pos,
                "dof_names": dof_names,
                "local_body_pos": None,
                "link_body_list": None,
            }

            save_data = {
                "robot_motion": robot_motion,
                "human_motion": human_motion,
            }

            with open(out_path, "wb") as f:
                pickle.dump(save_data, f)

    print(f"[OK] Done. Saved pkls under: {out_root}")


if __name__ == "__main__":
    main()
"""
python bvh_dir_retarget_multirobot_net_save.py \
  --bvh_dir ../lafan1/bvh_test\
  --out_dir ../retargeting_result/test_time/ \
  --ckpt net/ckpt_retarget_multi_best/best.pt \
  --robots_json worknet/robots_sepinputs.json \
  --clip_len 64 --stride 16 \
  --amp \
  --overwrite
"""
