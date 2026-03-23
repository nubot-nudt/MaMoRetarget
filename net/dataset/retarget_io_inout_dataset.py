# retarget_io_inout_dataset.py
# -*- coding: utf-8 -*-
"""
读取：
- 输入 pkl：包含 human_motion（用于组装 X_h）
- 标签 pkl：位于另外文件夹，文件名 = 输入 stem + "_refine".pkl，包含 robot_motion（用于 y）

输出（默认）：
- x_joint: (L, N, Dj)   例如 (64, 14, 12) = local_pos(3)+local_rot6d(6)+joint_vel(3)
- x_root : (L, Dr)      例如 (64, 12) = root_pos(3)+root_rot6d(6)+root_vel(3)
- m_h    : (M,)         例如 (13,)
- y_dof  : (L, DoF)     例如 (64, 29)
- y_root : (L, 9)       = root_pos(3)+root_rot6d(6)
- meta   : dict

注意：
- human_motion 的 local_* 是 root-relative，不含 global root 运动；因此我们显式加入 x_root。
- 训练时每个 pkl 是连续动作；Dataset 内部按 window 切片，绝不跨 pkl 边界。
"""

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


# ---------------------------
# Quaternion / rotation utils
# ---------------------------

def _normalize_quat_xyzw(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """q: (...,4) in xyzw"""
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(norm, eps)


def quat_xyzw_to_rot6d(q_xyzw: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (xyzw) -> rotation 6D (first 2 columns of rotation matrix).
    q_xyzw: (...,4)
    return: (...,6)
    """
    q = _normalize_quat_xyzw(np.asarray(q_xyzw, dtype=np.float32))
    x, y, z, w = np.split(q, 4, axis=-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    xw, yw, zw = x * w, y * w, z * w

    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - zw)
    r02 = 2 * (xz + yw)

    r10 = 2 * (xy + zw)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - xw)

    r20 = 2 * (xz - yw)
    r21 = 2 * (yz + xw)
    r22 = 1 - 2 * (xx + yy)

    # 6D: first two columns
    c1 = np.concatenate([r00, r10, r20], axis=-1)
    c2 = np.concatenate([r01, r11, r21], axis=-1)
    rot6d = np.concatenate([c1, c2], axis=-1).astype(np.float32)
    return rot6d


# ---------------------------
# IO helpers
# ---------------------------

def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path} is not a dict pkl.")
    return obj


def load_morph_npz(npz_path: str,
                   key_prefer: Tuple[str, ...] = ("target_edge_lengths", "bone_len")) -> np.ndarray:
    """
    读取静态 morphology 向量 m_h
    - 优先 target_edge_lengths (E,)
    - 否则 bone_len (Nfull,)
    """
    z = np.load(npz_path, allow_pickle=True)
    for k in key_prefer:
        if k in z.files:
            mh = np.asarray(z[k], dtype=np.float32).reshape(-1).copy()
            return mh
    raise KeyError(f"{npz_path} missing keys {key_prefer}. Available: {z.files}")


def resolve_label_pkl(input_pkl: str,
                      label_dir: str,
                      suffix: str = "_refine") -> str:
    """
    默认映射：
      input:  /.../aiming1_subject1.pkl
      label:  {label_dir}/aiming1_subject1_refine.pkl

    若精确文件不存在，则尝试 fuzzy：
      - label_dir 下所有以 stem 开头且包含 suffix 的 .pkl，若有多个取最短文件名。
    """
    stem = os.path.splitext(os.path.basename(input_pkl))[0]
    ext = os.path.splitext(input_pkl)[1]  # ".pkl"
    cand = os.path.join(label_dir, f"{stem}{suffix}{ext}")
    if os.path.exists(cand):
        return cand

    if not os.path.isdir(label_dir):
        raise FileNotFoundError(f"label_dir not found: {label_dir}")

    matches = []
    for fn in os.listdir(label_dir):
        if not fn.endswith(ext):
            continue
        if fn.startswith(stem) and (suffix in fn):
            matches.append(os.path.join(label_dir, fn))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        matches.sort(key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
        return matches[0]

    raise FileNotFoundError(
        f"Cannot find label pkl for input '{input_pkl}'. "
        f"Tried: {cand} and fuzzy pattern '{stem}*{suffix}*{ext}' in {label_dir}"
    )


# ---------------------------
# Feature builders (inputs / outputs)
# ---------------------------

def _diff_vel(x: np.ndarray, fps: float, vel_in_mps: bool) -> np.ndarray:
    """
    x: (T, ..., C)
    vel: (T, ..., C)
    - vel_in_mps=False: vel = x[t]-x[t-1]  (per-frame displacement)
    - vel_in_mps=True : vel = (x[t]-x[t-1]) * fps  (approx m/s)
    """
    vel = np.zeros_like(x, dtype=np.float32)
    vel[1:] = x[1:] - x[:-1]
    if vel_in_mps:
        vel *= float(fps)
    return vel.astype(np.float32)


def _infer_fps(human_motion: dict, fps_default: float) -> float:
    if isinstance(human_motion, dict) and ("fps" in human_motion):
        return float(human_motion["fps"])
    return float(fps_default)


def build_X_joint_from_human_motion(human_motion: dict,
                                    use_vel: bool = True,
                                    fps_default: float = 30.0,
                                    vel_in_mps: bool = False) -> Tuple[np.ndarray, List[str]]:
    """
    x_joint[t,j] = [local_pos(3), local_rot6d(6), local_vel(3?)]
    -> (T,N,12) if use_vel else (T,N,9)
    """
    if "local_body_pos" not in human_motion or "local_body_rot" not in human_motion:
        raise KeyError("human_motion must contain 'local_body_pos' and 'local_body_rot'")

    fps = _infer_fps(human_motion, fps_default=fps_default)

    pos = np.asarray(human_motion["local_body_pos"], dtype=np.float32)  # (T,N,3)
    rot = np.asarray(human_motion["local_body_rot"], dtype=np.float32)  # (T,N,4) xyzw
    names = human_motion.get("link_body_list", None)
    if names is None:
        names = [f"J{i}" for i in range(pos.shape[1])]

    T, N, _ = pos.shape
    rot6d = quat_xyzw_to_rot6d(rot.reshape(-1, 4)).reshape(T, N, 6)

    feats = [pos, rot6d]
    if use_vel:
        vel = _diff_vel(pos, fps=fps, vel_in_mps=vel_in_mps)  # (T,N,3)
        feats.append(vel)

    x_joint = np.concatenate(feats, axis=-1).astype(np.float32)
    return x_joint, list(names)


def build_X_root_from_human_motion(human_motion: dict,
                                   use_vel: bool = True,
                                   fps_default: float = 30.0,
                                   vel_in_mps: bool = False) -> np.ndarray:
    """
    x_root[t] = [root_pos(3), root_rot6d(6), root_vel(3?)]
    -> (T,12) if use_vel else (T,9)
    """
    if "root_pos" not in human_motion or "root_rot" not in human_motion:
        raise KeyError("human_motion must contain 'root_pos' and 'root_rot' for x_root")

    fps = _infer_fps(human_motion, fps_default=fps_default)

    root_pos = np.asarray(human_motion["root_pos"], dtype=np.float32)  # (T,3)
    root_rot = np.asarray(human_motion["root_rot"], dtype=np.float32)  # (T,4) xyzw
    root_rot6d = quat_xyzw_to_rot6d(root_rot)  # (T,6)

    feats = [root_pos, root_rot6d]
    if use_vel:
        root_vel = _diff_vel(root_pos, fps=fps, vel_in_mps=vel_in_mps)  # (T,3)
        feats.append(root_vel)

    x_root = np.concatenate(feats, axis=-1).astype(np.float32)
    return x_root


def build_y_from_robot_motion(robot_motion: dict,
                              dof_dim: int = 29) -> Tuple[np.ndarray, np.ndarray]:
    """
    标签：
      - y_root: (T,9) = root_pos(3) + root_rot6d(6)
      - y_dof : (T,dof_dim) = dof_pos
    """
    for k in ("root_pos", "root_rot", "dof_pos"):
        if k not in robot_motion:
            raise KeyError(f"robot_motion missing '{k}'")

    root_pos = np.asarray(robot_motion["root_pos"], dtype=np.float32)   # (T,3)
    root_rot = np.asarray(robot_motion["root_rot"], dtype=np.float32)   # (T,4) xyzw
    dof_pos = np.asarray(robot_motion["dof_pos"], dtype=np.float32)     # (T,29)

    if dof_pos.shape[1] != dof_dim:
        raise ValueError(f"dof_dim mismatch: expected {dof_dim}, got {dof_pos.shape[1]}")

    root_rot6d = quat_xyzw_to_rot6d(root_rot)  # (T,6)
    y_root = np.concatenate([root_pos, root_rot6d], axis=-1).astype(np.float32)  # (T,9)
    y_dof = dof_pos.astype(np.float32)  # (T,29)
    return y_root, y_dof


# ---------------------------
# Dataset
# ---------------------------

@dataclass
class SeqRecord:
    input_pkl: str
    label_pkl: str
    x_joint: np.ndarray   # (T,N,Dj)
    x_root: np.ndarray    # (T,Dr)
    y_root: np.ndarray    # (T,9)
    y_dof: np.ndarray     # (T,DoF)
    m_h: np.ndarray       # (M,)
    joint_names: List[str]


class MultiPklInOutWindowDataset(Dataset):
    """
    输入：多个 input pkls，每个是连续动作
    标签：在 label_dir 下，名称=stem + suffix（默认 _refine）
    输出：window clip，不跨 pkl 边界
    """

    def __init__(
        self,
        input_pkl_paths: Optional[List[str]] = None,
        input_dir: Optional[str] = None,
        label_dir: str = "",
        morph_npz: Union[str, Dict[str, str]] = "",
        clip_len: int = 64,
        stride: int = 16,
        suffix: str = "_refine",
        use_vel_joint: bool = True,
        use_vel_root: bool = True,
        vel_in_mps: bool = False,
        fps_default: float = 30.0,
        dof_dim: int = 29,
        align_mode: str = "min",      # "min" or "strict"
        device_tensor: bool = True,
    ):
        if torch is None:
            raise RuntimeError("PyTorch not found. Please install torch to use this Dataset.")

        # Backward-compatible convenience: allow passing a directory.
        if input_pkl_paths is None:
            if input_dir is None:
                raise TypeError("Must provide either input_pkl_paths or input_dir.")
            if not os.path.isdir(input_dir):
                raise FileNotFoundError(f"input_dir not found: {input_dir}")
            input_pkl_paths = [
                os.path.join(input_dir, fn)
                for fn in sorted(os.listdir(input_dir))
                if fn.endswith(".pkl")
            ]
        else:
            if input_dir is not None:
                raise TypeError("Provide only one of input_pkl_paths or input_dir, not both.")

        if not isinstance(input_pkl_paths, list) or len(input_pkl_paths) == 0:
            raise ValueError("input_pkl_paths is empty. Check your input_dir or file list.")

        self.clip_len = int(clip_len)
        self.stride = int(stride)
        self.device_tensor = bool(device_tensor)

        self.records: List[SeqRecord] = []
        self.index: List[Tuple[int, int]] = []  # (record_id, start_t)

        for in_pkl in input_pkl_paths:
            lb_pkl = resolve_label_pkl(in_pkl, label_dir=label_dir, suffix=suffix)

            in_data = load_pkl(in_pkl)
            lb_data = load_pkl(lb_pkl)

            if "human_motion" not in in_data:
                raise KeyError(f"{in_pkl} missing 'human_motion'. Keys={list(in_data.keys())}")
            if "robot_motion" not in lb_data:
                raise KeyError(f"{lb_pkl} missing 'robot_motion'. Keys={list(lb_data.keys())}")

            hm = in_data["human_motion"]
            x_joint, joint_names = build_X_joint_from_human_motion(
                hm,
                use_vel=use_vel_joint,
                fps_default=fps_default,
                vel_in_mps=vel_in_mps,
            )
            x_root = build_X_root_from_human_motion(
                hm,
                use_vel=use_vel_root,
                fps_default=fps_default,
                vel_in_mps=vel_in_mps,
            )

            y_root, y_dof = build_y_from_robot_motion(lb_data["robot_motion"], dof_dim=dof_dim)

            # 对齐长度
            T_in = x_joint.shape[0]
            T_root_in = x_root.shape[0]
            T_lb = y_dof.shape[0]
            T_lb_root = y_root.shape[0]

            if not (T_in == T_root_in):
                raise ValueError(f"{in_pkl}: x_joint T={T_in} != x_root T={T_root_in} (human_motion inconsistent?)")

            if not (T_lb == T_lb_root):
                raise ValueError(f"{lb_pkl}: y_dof T={T_lb} != y_root T={T_lb_root} (robot_motion inconsistent?)")

            if align_mode == "strict":
                if T_in != T_lb:
                    raise ValueError(f"T mismatch strict: {in_pkl} T={T_in} vs {lb_pkl} T={T_lb}")
                T = T_in
            elif align_mode == "min":
                T = min(T_in, T_lb)
                x_joint = x_joint[:T]
                x_root = x_root[:T]
                y_root = y_root[:T]
                y_dof = y_dof[:T]
            else:
                raise ValueError("align_mode must be 'min' or 'strict'.")

            # morphology（静态）
            motion_id = os.path.splitext(os.path.basename(in_pkl))[0]
            if isinstance(morph_npz, str):
                mh = load_morph_npz(morph_npz)
            else:
                if motion_id not in morph_npz:
                    raise KeyError(f"morph_npz dict missing key '{motion_id}' for {in_pkl}")
                mh = load_morph_npz(morph_npz[motion_id])

            rec = SeqRecord(
                input_pkl=in_pkl,
                label_pkl=lb_pkl,
                x_joint=x_joint,
                x_root=x_root,
                y_root=y_root,
                y_dof=y_dof,
                m_h=mh,
                joint_names=joint_names,
            )
            rec_id = len(self.records)
            self.records.append(rec)

            # 生成窗口 index（不跨序列）
            if T < self.clip_len:
                continue
            for s in range(0, T - self.clip_len + 1, self.stride):
                self.index.append((rec_id, s))

        if len(self.index) == 0:
            raise RuntimeError("No valid windows created. Check clip_len/stride and sequence lengths.")

    def __len__(self) -> int:
        return len(self.index)

    def summary(self) -> Dict[str, object]:
        """Return dataset statistics helpful for debugging."""
        import collections

        per_rec = collections.Counter([rid for rid, _ in self.index])
        seq_stats = []
        for rid, rec in enumerate(self.records):
            T = int(rec.x_joint.shape[0])
            n_win = int(per_rec.get(rid, 0))
            if n_win > 0:
                last_s = max(s for (rr, s) in self.index if rr == rid)
                covered = last_s + self.clip_len
                dropped = max(0, T - covered)
            else:
                dropped = T
            seq_stats.append({
                "record_id": rid,
                "input_pkl": rec.input_pkl,
                "label_pkl": rec.label_pkl,
                "T": T,
                "n_windows": n_win,
                "dropped_tail_frames": int(dropped),
            })

        return {
            "num_records": len(self.records),
            "num_windows": len(self.index),
            "clip_len": self.clip_len,
            "stride": self.stride,
            "sequences": seq_stats,
        }

    def __getitem__(self, idx: int):
        rec_id, s = self.index[idx]
        rec = self.records[rec_id]
        e = s + self.clip_len

        xj = rec.x_joint[s:e]   # (L,N,Dj)
        xr = rec.x_root[s:e]    # (L,Dr)
        mh = rec.m_h            # (M,)
        yd = rec.y_dof[s:e]     # (L,DoF)
        yr = rec.y_root[s:e]    # (L,9)

        meta = {
            "input_pkl": rec.input_pkl,
            "label_pkl": rec.label_pkl,
            "start": s,
            "end": e,
            "joint_names": rec.joint_names,
        }

        if self.device_tensor:
            xj = torch.from_numpy(xj)
            xr = torch.from_numpy(xr)
            mh = torch.from_numpy(mh)
            yd = torch.from_numpy(yd)
            yr = torch.from_numpy(yr)

        return xj, xr, mh, yd, yr, meta


if __name__ == "__main__":
    import argparse
    from pprint import pprint

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="dataset/input")
    parser.add_argument("--label_dir", type=str, default="dataset/output")
    parser.add_argument("--morph_npz", type=str, default="morph_out/human_morphology_targets.npz")
    parser.add_argument("--clip_len", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--suffix", type=str, default="_refine")
    parser.add_argument("--no_vel", action="store_true", help="disable joint/root velocity features")
    parser.add_argument("--vel_in_mps", action="store_true")
    parser.add_argument("--fps_default", type=float, default=30.0)
    parser.add_argument("--align_mode", type=str, default="min", choices=["min", "strict"])
    parser.add_argument("--cpu", action="store_true", help="return numpy (not torch tensors)")
    args = parser.parse_args()

    ds = MultiPklInOutWindowDataset(
        input_dir=args.input_dir,
        label_dir=args.label_dir,
        morph_npz=args.morph_npz,
        clip_len=args.clip_len,
        stride=args.stride,
        suffix=args.suffix,
        use_vel_joint=(not args.no_vel),
        use_vel_root=(not args.no_vel),
        vel_in_mps=bool(args.vel_in_mps),
        fps_default=float(args.fps_default),
        dof_dim=29,
        align_mode=args.align_mode,
        device_tensor=(not args.cpu),
    )

    print(f"#windows = {len(ds)}")
    pprint(ds.summary())

    xj, xr, mh, yd, yr, meta = ds[0]
    print("x_joint:", tuple(xj.shape), "x_root:", tuple(xr.shape), "m_h:", tuple(mh.shape),
          "y_dof:", tuple(yd.shape), "y_root:", tuple(yr.shape))
    print("meta:")
    pprint(meta)
