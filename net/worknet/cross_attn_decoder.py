# cross_attn_decoder.py
# -*- coding: utf-8 -*-
"""
Cross-attention decoder for retargeting:
  - Query:  Robot joint tokens R (Nr, D)
  - Key/Val: Human tokens H_all (B, L, N, D) where N = Nh + 1 (token0 is human root)

Outputs:
  - y_dof  : (B, L, Nr)
  - y_root : (B, L, root_out_dim)  (default 9 = pos3 + rot6d; you can set 7 = pos3 + quat4)
  - attn   : (B, L, Nr, N)  (optional, for debugging or M_joint regularization)

This file can be imported as a module, or run directly for a quick sanity check.

Usage (self-test):
  python cross_attn_decoder.py --robot_spec robot_spec_g1.npz --B 2 --L 64 --N 15 --Nr 29 --D 128
"""

from __future__ import annotations

import argparse
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------
def load_robot_spec_npz(npz_path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load robot_spec.npz and return a dict of torch tensors (only numeric fields).
    At minimum, decoder expects: q_min, q_max, limited, (optional rest_q)
    """
    z = np.load(npz_path, allow_pickle=True)
    spec: Dict[str, torch.Tensor] = {}
    for k in z.files:
        v = z[k]
        # keep only numeric arrays
        if isinstance(v, np.ndarray) and v.dtype != object:
            spec[k] = torch.as_tensor(v, device=device)
    return spec


def _ensure_robot_spec_min_fields(robot_spec: Dict[str, torch.Tensor], Nr: int, device: torch.device):
    """
    Ensure q_min/q_max/limited/rest_q exist with correct shapes.
    If missing, create safe defaults.
    """
    if "q_min" not in robot_spec:
        robot_spec["q_min"] = torch.full((Nr,), -torch.pi, device=device)
    if "q_max" not in robot_spec:
        robot_spec["q_max"] = torch.full((Nr,), torch.pi, device=device)
    if "limited" not in robot_spec:
        robot_spec["limited"] = torch.ones((Nr,), device=device)  # assume limited
    if "rest_q" not in robot_spec:
        robot_spec["rest_q"] = torch.zeros((Nr,), device=device)

    # shape check / fix
    for k in ["q_min", "q_max", "limited", "rest_q"]:
        robot_spec[k] = robot_spec[k].to(device).view(-1)
        if robot_spec[k].numel() != Nr:
            raise ValueError(f"robot_spec['{k}'] has shape {tuple(robot_spec[k].shape)} but Nr={Nr} expected.")


# -----------------------------
# Decoder
# -----------------------------
class CrossAttnRetargetDecoder(nn.Module):
    """
    Cross-attn decoder:
      - For each frame t, robot joint tokens query human tokens to produce context per robot joint.
      - Then a small head predicts joint angle (scalar) per robot joint.
      - Optionally predicts robot root using a learned query token.

    Shapes:
      H_all: (B, L, N, D)
      R:     (Nr, D)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        root_out_dim: int = 9,     # 9=pos3+rot6d, or set 7=pos3+quat4
        use_root_head: bool = True,
        limit_mode: str = "tanh",  # "tanh" or "none"
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert limit_mode in ("tanh", "none")

        self.d_model = d_model
        self.n_heads = n_heads
        self.root_out_dim = root_out_dim
        self.use_root_head = use_root_head
        self.limit_mode = limit_mode

        # Robot->Human cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # DOF head: (B,L,Nr,D) -> (B,L,Nr)
        self.dof_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Root head (optional)
        if use_root_head:
            self.root_query = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.root_query, std=0.02)

            self.root_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.root_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, root_out_dim),
            )

    @staticmethod
    def apply_joint_limits(raw: torch.Tensor, robot_spec: Dict[str, torch.Tensor], mode: str = "tanh") -> torch.Tensor:
        """
        raw: (B, L, Nr) unconstrained
        robot_spec must contain q_min,q_max,limited,rest_q
        mode:
          - "tanh": limited joints are squashed into [qmin,qmax]
          - "none": return raw + rest_q
        """
        qmin = robot_spec["q_min"].float()       # (Nr,)
        qmax = robot_spec["q_max"].float()       # (Nr,)
        limited = robot_spec["limited"].float()  # (Nr,) 0/1
        rest_q = robot_spec["rest_q"].float()    # (Nr,)

        if mode == "none":
            return raw + rest_q.view(1, 1, -1)

        # center/range
        center = 0.5 * (qmax + qmin)
        half = 0.5 * (qmax - qmin)

        squashed = center.view(1, 1, -1) + torch.tanh(raw) * half.view(1, 1, -1)
        raw_bias = raw + rest_q.view(1, 1, -1)

        out = limited.view(1, 1, -1) * squashed + (1.0 - limited.view(1, 1, -1)) * raw_bias
        return out

    def forward(
        self,
        H_all: torch.Tensor,                      # (B,L,N,D)
        R: torch.Tensor,                          # (Nr,D)
        robot_spec: Dict[str, torch.Tensor],
        return_attn: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
          y_dof:  (B,L,Nr)
          y_root: (B,L,root_out_dim) or None
          attn:   (B,L,Nr,N) or None
        """
        assert H_all.dim() == 4, f"H_all must be (B,L,N,D), got {tuple(H_all.shape)}"
        assert R.dim() == 2, f"R must be (Nr,D), got {tuple(R.shape)}"

        B, L, N, D = H_all.shape
        Nr, Dr = R.shape
        assert Dr == D, f"Token dim mismatch: H_all last dim {D}, R last dim {Dr}"

        device = H_all.device
        _ensure_robot_spec_min_fields(robot_spec, Nr=Nr, device=device)

        # Flatten time into batch for efficiency: (B,L,...) -> (B*L,...)
        K = H_all.reshape(B * L, N, D)                    # (B*L, N, D)
        V = K

        # Expand robot tokens to each frame as queries
        Q = R.to(device).unsqueeze(0).expand(B * L, Nr, D)  # (B*L, Nr, D)

        # cross-attn
        out, attn = self.cross_attn(
            query=Q,
            key=K,
            value=V,
            need_weights=return_attn,
            average_attn_weights=True,  # (B*L, Nr, N)
        )
        out = out.reshape(B, L, Nr, D)                    # (B,L,Nr,D)

        attn_out = None
        if return_attn:
            attn_out = attn.reshape(B, L, Nr, N)          # (B,L,Nr,N)

        # DOF prediction
        raw = self.dof_head(out).squeeze(-1)              # (B,L,Nr)
        y_dof = self.apply_joint_limits(raw, robot_spec, mode=self.limit_mode)

        # Root prediction (optional)
        y_root = None
        if self.use_root_head:
            rq = self.root_query.to(device).expand(B * L, 1, D)          # (B*L,1,D)
            rctx, _ = self.root_attn(rq, K, V, need_weights=False)       # (B*L,1,D)
            rctx = rctx.reshape(B, L, D)                                 # (B,L,D)
            y_root = self.root_head(rctx)                                # (B,L,root_out_dim)

        return y_dof, y_root, attn_out


# -----------------------------
# Optional: M_joint regularizer
# -----------------------------
def mjoint_attention_regularizer(
    attn: torch.Tensor,            # (B,L,Nr,N)
    M_joint: torch.Tensor,         # (Nr,Nh) or (Nr,N)
    eps: float = 1e-6,
    include_root_token: bool = False,
) -> torch.Tensor:
    """
    Encourage robot->human attention distribution to match prior mapping M_joint.
    Use KL(attn || M).

    If include_root_token=False, drop human root token (index 0) and expect M_joint has Nh columns.
    """
    if attn is None:
        raise ValueError("attn is None. Call decoder with return_attn=True.")

    if not include_root_token:
        attn_use = attn[..., 1:]        # (B,L,Nr,Nh)
    else:
        attn_use = attn                 # (B,L,Nr,N)

    attn_p = attn_use.clamp_min(eps)
    attn_p = attn_p / attn_p.sum(dim=-1, keepdim=True)

    M = M_joint.to(attn.device).float().clamp_min(eps)
    M = M / M.sum(dim=-1, keepdim=True)

    # broadcast
    M = M.view(1, 1, M.shape[0], M.shape[1]).expand_as(attn_p)

    kl = (attn_p * (attn_p.log() - M.log())).sum(dim=-1)  # (B,L,Nr)
    return kl.mean()


# -----------------------------
# Self-test main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_spec", type=str, default="", help="Path to robot_spec.npz (optional for test)")
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--N", type=int, default=15, help="Human token count (Nh+1)")
    ap.add_argument("--Nr", type=int, default=29)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--root_out_dim", type=int, default=9, help="9=pos3+rot6d, 7=pos3+quat4")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--no_root", action="store_true")
    ap.add_argument("--no_attn", action="store_true")
    ap.add_argument("--limit_mode", type=str, default="tanh", choices=["tanh", "none"])
    args = ap.parse_args()

    device = torch.device(args.device)

    # Dummy inputs
    H_all = torch.randn(args.B, args.L, args.N, args.D, device=device)
    R = torch.randn(args.Nr, args.D, device=device)

    # robot_spec
    if args.robot_spec:
        robot_spec = load_robot_spec_npz(args.robot_spec, device=args.device)
    else:
        # Minimal defaults
        robot_spec = {
            "q_min": torch.full((args.Nr,), -torch.pi, device=device),
            "q_max": torch.full((args.Nr,), torch.pi, device=device),
            "limited": torch.ones((args.Nr,), device=device),
            "rest_q": torch.zeros((args.Nr,), device=device),
        }

    dec = CrossAttnRetargetDecoder(
        d_model=args.D,
        n_heads=args.heads,
        dropout=0.1,
        root_out_dim=args.root_out_dim,
        use_root_head=(not args.no_root),
        limit_mode=args.limit_mode,
    ).to(device)

    y_dof, y_root, attn = dec(H_all, R, robot_spec, return_attn=(not args.no_attn))

    print("[OK] y_dof:", tuple(y_dof.shape))
    print("[OK] y_root:", None if y_root is None else tuple(y_root.shape))
    print("[OK] attn  :", None if attn is None else tuple(attn.shape))
    print("[range] y_dof min/mean/max:",
          float(y_dof.min().item()), float(y_dof.mean().item()), float(y_dof.max().item()))

    if y_root is not None:
        print("[range] y_root min/mean/max:",
              float(y_root.min().item()), float(y_root.mean().item()), float(y_root.max().item()))


if __name__ == "__main__":
    main()
