# train_retarget_multirobot_sepinputs.py
# -*- coding: utf-8 -*-

"""Multi-robot conditional retargeting trainer (per-robot human inputs).

This is a variant of train_retarget_multirobot.py.

Key difference:
  - Each robot can have its OWN *scaled* human input pkls (train/val), instead of sharing one global input folder.

Why you might want this:
  - You pre-scale human motions to each robot's morphology/scale offline for convenience and stability.
  - Then each robot has (train_input_dir, val_input_dir) and matching label dirs.

Important constraint (still required):
  - We STILL cannot mix different robots inside the same batch because their DOF dimension Nr differs.

robots_json format
-----------------
List (or dict with key "robots") of entries like:

  {
    "name": "g1",
    "robot_spec_npz": "worknet/robot_spec_g1.npz",

    "train_input_dir": "dataset_g1/train/input_scaled",
    "val_input_dir":   "dataset_g1/val/input_scaled",

    "train_label_dir": "dataset_g1/train/output",
    "val_label_dir":   "dataset_g1/val/output",

    "morph_npz": "morph_out/human_morphology_targets_g1.npz",   // optional; fallback to CLI --morph_npz

    "dof_dim": 29,           // optional (infer from robot_spec if missing)
    "suffix": "_refine",    // optional (default from CLI)
    "weight": 1.0            // optional sampling weight for mix_mode=weighted
  }

CLI fallbacks:
  - If a robot entry misses train_input_dir/val_input_dir, we fallback to --train_input_dir/--val_input_dir.
  - If a robot entry misses morph_npz, we fallback to --morph_npz.

Run example
-----------
python train_retarget_multirobot_sepinputs.py \
  --robots_json worknet/robots_sepinputs.json \
  --out_dir net/ckpt_retarget_multi_new \
  --clip_len 64 --stride 16 \
  --min_lr 1e-6 \
  --batch_size 16 --epochs 500 --lr 3e-4 \
  --amp  \
  --mix_mode per_robot

python train_retarget_multirobot_sepinputs.py \
  --robots_json worknet/robots_g1_n1.json \
  --out_dir net/ckpt_retarget_g1_n1_300 \
  --clip_len 64 --stride 16 \
  --batch_size 16 --epochs 300 --lr 3e-4 \
  --min_lr 1e-6 \
  --amp \
  --mix_mode per_robot
"""

import os
import math
import time
import json
import random
import argparse
from typing import Dict, List, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# keep import style consistent with your repo
from dataset.retarget_io_inout_dataset import MultiPklInOutWindowDataset
from worknet.human_encoder import HumanEncoder
from worknet.robot_spec_encoder import load_robot_spec_npz, build_robot_node_features, RobotSpecEncoder
from worknet.cross_attn_decoder import CrossAttnRetargetDecoder


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_pkls(folder: str) -> List[str]:
    if not folder:
        return []
    if not os.path.isdir(folder):
        raise RuntimeError(f"Not a folder: {folder}")
    return [
        os.path.join(folder, fn)
        for fn in sorted(os.listdir(folder))
        if fn.endswith(".pkl")
    ]


def collate_keep_meta_with_robot(batch):
    """Batch: (xj, xr, mh, yd, yr, robot_name, meta)."""
    xj, xr, mh, yd, yr, rname, meta = zip(*batch)
    return (
        torch.stack(xj, 0),
        torch.stack(xr, 0),
        torch.stack(mh, 0),
        torch.stack(yd, 0),
        torch.stack(yr, 0),
        list(rname),
        list(meta),
    )


class _RobotWrappedDataset(torch.utils.data.Dataset):
    """Wrap a MultiPklInOutWindowDataset to additionally return a fixed robot_name."""

    def __init__(self, base_ds: torch.utils.data.Dataset, robot_name: str):
        self.base_ds = base_ds
        self.robot_name = robot_name

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        xj, xr, mh, yd, yr, meta = self.base_ds[idx]
        return xj, xr, mh, yd, yr, self.robot_name, meta


def load_robots_json(path: str) -> List[Dict[str, Any]]:
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

    out: List[Dict[str, Any]] = []
    for r in robots:
        if not isinstance(r, dict):
            raise RuntimeError("Each robot entry must be a dict")
        if "name" not in r:
            raise RuntimeError("Each robot entry must have 'name'")
        if "robot_spec_npz" not in r:
            raise RuntimeError(f"Robot {r.get('name')} missing robot_spec_npz")
        out.append(r)
    return out


def maybe_infer_dof_dim(robot_spec: Dict[str, torch.Tensor]) -> int:
    if "q_min" in robot_spec:
        return int(robot_spec["q_min"].numel())
    if "joint_names" in robot_spec and hasattr(robot_spec["joint_names"], "__len__"):
        return int(len(robot_spec["joint_names"]))
    raise RuntimeError("Cannot infer dof_dim from robot_spec. Please set dof_dim in robots_json.")


def compute_total_train_steps(loaders: Dict[str, DataLoader], epochs: int) -> int:
    steps_per_epoch = sum(len(ld) for ld in loaders.values())
    return max(1, steps_per_epoch * max(1, epochs))



def cosine_lr(step, total_steps, base_lr, min_lr=0.0)-> float:
    total_steps = max(int(total_steps), 1)
    den = max(total_steps - 1, 1)
    step = min(max(int(step), 0), den)
    cos = 0.5 * (1.0 + math.cos(math.pi * step / float(den)))
    return float(min_lr + (base_lr - min_lr) * cos)

# -----------------------------
# Eval
# -----------------------------

@torch.no_grad()
def eval_one_robot(
    human_encoder: HumanEncoder,
    robot_encoder: RobotSpecEncoder,
    decoder: CrossAttnRetargetDecoder,
    loader: DataLoader,
    robot_spec: Dict[str, torch.Tensor],
    node_feat: torch.Tensor,
    device: torch.device,
    w_root: float,
    freeze_robot_encoder: bool,
    cached_R: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    human_encoder.eval()
    robot_encoder.eval()
    decoder.eval()

    loss_sum = loss_dof_sum = loss_root_sum = 0.0
    n = 0

    for xj, xr, mh, yd, yr, rnames, meta in loader:
        xj = xj.to(device, non_blocking=True)
        xr = xr.to(device, non_blocking=True)
        mh = mh.to(device, non_blocking=True)
        yd = yd.to(device, non_blocking=True)
        yr = yr.to(device, non_blocking=True)

        H_all, _, _ = human_encoder(xj, xr, mh)
        if freeze_robot_encoder and cached_R is not None:
            R = cached_R
        else:
            R = robot_encoder(node_feat, robot_spec["parent_idx"], robot_spec["joint_type"])

        y_dof_pred, y_root_pred, _ = decoder(H_all, R, robot_spec, return_attn=False)

        loss_dof = F.mse_loss(y_dof_pred, yd)
        loss_root = F.mse_loss(y_root_pred, yr)
        loss = loss_dof + w_root * loss_root

        bs = xj.shape[0]
        n += bs
        loss_sum += float(loss.item()) * bs
        loss_dof_sum += float(loss_dof.item()) * bs
        loss_root_sum += float(loss_root.item()) * bs

    return {
        "loss": loss_sum / max(n, 1),
        "loss_dof": loss_dof_sum / max(n, 1),
        "loss_root": loss_root_sum / max(n, 1),
    }


# -----------------------------
# Train
# -----------------------------

def _resolve_robot_path(r: Dict[str, Any], key: str, fallback: Optional[str]) -> Optional[str]:
    v = r.get(key, None)
    if v is None or str(v).strip() == "":
        return fallback
    return str(v)


def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    robots_cfg = load_robots_json(args.robots_json)

    # ------------------------
    # Build per-robot datasets/loaders/specs
    # ------------------------
    robot_specs: Dict[str, Dict[str, torch.Tensor]] = {}
    robot_node_feat: Dict[str, torch.Tensor] = {}
    robot_weight: Dict[str, float] = {}
    train_loaders: Dict[str, DataLoader] = {}
    val_loaders: Dict[str, DataLoader] = {}
    robot_train_counts: Dict[str, int] = {}
    robot_val_counts: Dict[str, int] = {}

    shared_dims = None  # (Nh,Dj,Dr,morph_dim,root_dim)

    for r in robots_cfg:
        name = str(r["name"])
        spec_npz = str(r["robot_spec_npz"])

        # per-robot input dirs (scaled human motions)
        train_input_dir = _resolve_robot_path(r, "train_input_dir", args.train_input_dir)
        val_input_dir = _resolve_robot_path(r, "val_input_dir", args.val_input_dir)
        if not train_input_dir or not val_input_dir:
            raise RuntimeError(
                f"Robot {name} missing train_input_dir/val_input_dir. "
                f"Provide them in robots_json or via CLI --train_input_dir/--val_input_dir."
            )
        train_paths = list_pkls(train_input_dir)
        val_paths = list_pkls(val_input_dir)
        if len(train_paths) == 0:
            raise RuntimeError(f"Robot {name}: no .pkl found in train_input_dir={train_input_dir}")
        if len(val_paths) == 0:
            raise RuntimeError(f"Robot {name}: no .pkl found in val_input_dir={val_input_dir}")

        # per-robot label dirs
        train_label_dir = r.get("train_label_dir", None)
        val_label_dir = r.get("val_label_dir", None)
        if not train_label_dir or not val_label_dir:
            raise RuntimeError(f"Robot {name} must provide train_label_dir and val_label_dir in robots_json.")

        # per-robot morph (optional)
        morph_npz = _resolve_robot_path(r, "morph_npz", args.morph_npz)
        if not morph_npz:
            raise RuntimeError(
                f"Robot {name} missing morph_npz. Provide in robots_json or via CLI --morph_npz."
            )

        suffix = str(r.get("suffix", args.suffix))
        weight = float(r.get("weight", 1.0))

        # load robot spec on device
        spec = load_robot_spec_npz(spec_npz, device=str(device))
        dof_dim = int(r.get("dof_dim", maybe_infer_dof_dim(spec)))

        # dataset (train/val)
        ds_train_base = MultiPklInOutWindowDataset(
            input_pkl_paths=train_paths,
            label_dir=train_label_dir,
            morph_npz=morph_npz,
            clip_len=args.clip_len,
            stride=args.stride,
            suffix=suffix,
            use_vel_joint=(not args.no_vel),
            use_vel_root=(not args.no_vel),
            vel_in_mps=args.vel_in_mps,
            fps_default=args.fps_default,
            dof_dim=dof_dim,
            align_mode=args.align_mode,
            device_tensor=True,
        )
        ds_val_base = MultiPklInOutWindowDataset(
            input_pkl_paths=val_paths,
            label_dir=val_label_dir,
            morph_npz=morph_npz,
            clip_len=args.clip_len,
            stride=args.stride,
            suffix=suffix,
            use_vel_joint=(not args.no_vel),
            use_vel_root=(not args.no_vel),
            vel_in_mps=args.vel_in_mps,
            fps_default=args.fps_default,
            dof_dim=dof_dim,
            align_mode=args.align_mode,
            device_tensor=True,
        )

        ds_train = _RobotWrappedDataset(ds_train_base, robot_name=name)
        ds_val = _RobotWrappedDataset(ds_val_base, robot_name=name)

        loader_train = DataLoader(
            ds_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            collate_fn=collate_keep_meta_with_robot,
            pin_memory=(device.type == "cuda"),
        )
        loader_val = DataLoader(
            ds_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            collate_fn=collate_keep_meta_with_robot,
            pin_memory=(device.type == "cuda"),
        )

        # check shared dims
        xj0, xr0, mh0, yd0, yr0, _, _ = ds_train[0]
        _, Nh0, Dj0 = xj0.shape
        Dr0 = xr0.shape[-1]
        morph_dim0 = mh0.shape[-1]
        root_dim0 = yr0.shape[-1]

        dims0 = (Nh0, Dj0, Dr0, morph_dim0, root_dim0)
        if shared_dims is None:
            shared_dims = dims0
        elif dims0 != shared_dims:
            raise RuntimeError(
                f"Shared human/root dims mismatch for robot {name}: got {dims0} but expected {shared_dims}.\n"
                "Hint: multi-robot training assumes SAME human feature construction (Nh/Dj/Dr) and same root dim."
            )

        node_feat = build_robot_node_features(spec, use_offset_vec=True, use_limits=True).to(device)

        robot_specs[name] = spec
        robot_node_feat[name] = node_feat
        robot_weight[name] = weight
        train_loaders[name] = loader_train
        val_loaders[name] = loader_val
        robot_train_counts[name] = len(ds_train)
        robot_val_counts[name] = len(ds_val)

        print(
            f"[Robot {name}] train_input_dir={train_input_dir} ({len(train_paths)} pkls) "
            f"val_input_dir={val_input_dir} ({len(val_paths)} pkls) | "
            f"train_windows={len(ds_train)} val_windows={len(ds_val)} | dof_dim={dof_dim}"
        )

    assert shared_dims is not None
    Nh, Dj, Dr, morph_dim, root_dim = shared_dims

    # ------------------------
    # Build shared models
    # ------------------------
    human_encoder = HumanEncoder(
        Nh=Nh,
        Dj=Dj,
        Dr=Dr,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.human_layers,
        dropout=args.dropout,
        max_len=max(1024, args.clip_len),
        use_morph_film=(not args.no_morph_film),
        morph_dim=morph_dim,
    ).to(device)

    any_robot = next(iter(robot_node_feat.keys()))
    node_feat_dim = int(robot_node_feat[any_robot].shape[-1])
    robot_encoder = RobotSpecEncoder(
        node_feat_dim=node_feat_dim,
        d_model=args.d_model,
        num_layers=args.robot_layers,
        dropout=args.dropout,
    ).to(device)

    decoder = CrossAttnRetargetDecoder(
        d_model=args.d_model,
        n_heads=args.heads,
        dropout=args.dropout,
        root_out_dim=root_dim,
        use_root_head=True,
        limit_mode=args.limit_mode,
    ).to(device)

    if args.freeze_robot_encoder:
        for p in robot_encoder.parameters():
            p.requires_grad_(False)

    params = list(human_encoder.parameters()) + list(decoder.parameters())
    if not args.freeze_robot_encoder:
        params += list(robot_encoder.parameters())

    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    total_steps = compute_total_train_steps(train_loaders, args.epochs)
    global_step = 0

    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    cached_R: Dict[str, torch.Tensor] = {}
    if args.freeze_robot_encoder:
        robot_encoder.eval()
        with torch.no_grad():
            for name, spec in robot_specs.items():
                R = robot_encoder(robot_node_feat[name], spec["parent_idx"], spec["joint_type"]).detach()
                cached_R[name] = R

    best_val = float("inf")

    # save config
    cfg_path = os.path.join(args.out_dir, "train_config_multirobot_sepinputs.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(f"[Robots] {list(robot_specs.keys())}")
    print(f"[Shared dims] Nh={Nh} Dj={Dj} Dr={Dr} morph={morph_dim} root_dim={root_dim} d_model={args.d_model}")
    print(f"[Device] {device.type} | AMP={scaler.is_enabled()} | freeze_robot_encoder={args.freeze_robot_encoder}")

    for epoch in range(args.epochs):
        t0 = time.time()

        human_encoder.train()
        decoder.train()
        robot_encoder.eval() if args.freeze_robot_encoder else robot_encoder.train()

        tr_loss = tr_dof = tr_root = 0.0
        tr_n = 0

        robot_names = list(train_loaders.keys())

        if args.mix_mode == "per_robot":
            for rname in robot_names:
                loader = train_loaders[rname]
                spec = robot_specs[rname]
                node_feat = robot_node_feat[rname]
                R_cache = cached_R.get(rname, None)

                for xj, xr, mh, yd, yr, _, _ in loader:
                    xj = xj.to(device, non_blocking=True)
                    xr = xr.to(device, non_blocking=True)
                    mh = mh.to(device, non_blocking=True)
                    yd = yd.to(device, non_blocking=True)
                    yr = yr.to(device, non_blocking=True)

                    if args.lr_schedule == "cosine":
                        lr_now = cosine_lr(global_step, total_steps, args.lr, min_lr=args.min_lr)
                        for pg in optim.param_groups:
                            pg["lr"] = lr_now
                    else:
                        lr_now = optim.param_groups[0]["lr"]

                    optim.zero_grad(set_to_none=True)
                    use_autocast = scaler.is_enabled()
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_autocast):
                        H_all, _, _ = human_encoder(xj, xr, mh)
                        if args.freeze_robot_encoder and R_cache is not None:
                            R = R_cache
                        else:
                            R = robot_encoder(node_feat, spec["parent_idx"], spec["joint_type"])

                        y_dof_pred, y_root_pred, _ = decoder(H_all, R, spec, return_attn=False)
                        loss_dof = F.mse_loss(y_dof_pred, yd)
                        loss_root = F.mse_loss(y_root_pred, yr)
                        loss = loss_dof + args.w_root * loss_root

                    scaler.scale(loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                    scaler.step(optim)
                    scaler.update()

                    bs = xj.shape[0]
                    tr_n += bs
                    tr_loss += float(loss.item()) * bs
                    tr_dof += float(loss_dof.item()) * bs
                    tr_root += float(loss_root.item()) * bs

                    global_step += 1
                    if (global_step % args.log_every) == 0:
                        print(
                            f"[E{epoch:03d} step{global_step:06d}] robot={rname} "
                            f"loss={loss.item():.6f} dof={loss_dof.item():.6f} root={loss_root.item():.6f} lr={lr_now:.2e}"
                        )

        else:
            # roundrobin / weighted: fixed steps per epoch
            steps_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else sum(len(train_loaders[n]) for n in robot_names)
            iters = {n: iter(train_loaders[n]) for n in robot_names}
            weights = np.array([robot_weight[n] for n in robot_names], dtype=np.float64)
            weights = weights / max(weights.sum(), 1e-12)

            for _ in range(steps_epoch):
                if args.mix_mode == "roundrobin":
                    rname = robot_names[global_step % len(robot_names)]
                else:
                    rname = str(np.random.choice(robot_names, p=weights))

                try:
                    batch = next(iters[rname])
                except StopIteration:
                    iters[rname] = iter(train_loaders[rname])
                    batch = next(iters[rname])

                xj, xr, mh, yd, yr, _, _ = batch
                xj = xj.to(device, non_blocking=True)
                xr = xr.to(device, non_blocking=True)
                mh = mh.to(device, non_blocking=True)
                yd = yd.to(device, non_blocking=True)
                yr = yr.to(device, non_blocking=True)

                spec = robot_specs[rname]
                node_feat = robot_node_feat[rname]
                R_cache = cached_R.get(rname, None)

                if args.lr_schedule == "cosine":
                    lr_now = cosine_lr(global_step, total_steps, args.lr, min_lr=args.min_lr)
                    for pg in optim.param_groups:
                        pg["lr"] = lr_now
                else:
                    lr_now = optim.param_groups[0]["lr"]

                optim.zero_grad(set_to_none=True)
                use_autocast = scaler.is_enabled()
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_autocast):
                    H_all, _, _ = human_encoder(xj, xr, mh)
                    if args.freeze_robot_encoder and R_cache is not None:
                        R = R_cache
                    else:
                        R = robot_encoder(node_feat, spec["parent_idx"], spec["joint_type"])
                    y_dof_pred, y_root_pred, _ = decoder(H_all, R, spec, return_attn=False)
                    loss_dof = F.mse_loss(y_dof_pred, yd)
                    loss_root = F.mse_loss(y_root_pred, yr)
                    loss = loss_dof + args.w_root * loss_root

                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                scaler.step(optim)
                scaler.update()

                bs = xj.shape[0]
                tr_n += bs
                tr_loss += float(loss.item()) * bs
                tr_dof += float(loss_dof.item()) * bs
                tr_root += float(loss_root.item()) * bs

                global_step += 1
                if (global_step % args.log_every) == 0:
                    print(
                        f"[E{epoch:03d} step{global_step:06d}] robot={rname} "
                        f"loss={loss.item():.6f} dof={loss_dof.item():.6f} root={loss_root.item():.6f} lr={lr_now:.2e}"
                    )

        tr_loss /= max(tr_n, 1)
        tr_dof /= max(tr_n, 1)
        tr_root /= max(tr_n, 1)

        # eval per robot
        val_by_robot: Dict[str, Dict[str, float]] = {}
        val_loss_weighted = 0.0
        wsum = 0.0
        for rname, vloader in val_loaders.items():
            v = eval_one_robot(
                human_encoder=human_encoder,
                robot_encoder=robot_encoder,
                decoder=decoder,
                loader=vloader,
                robot_spec=robot_specs[rname],
                node_feat=robot_node_feat[rname],
                device=device,
                w_root=args.w_root,
                freeze_robot_encoder=args.freeze_robot_encoder,
                cached_R=cached_R.get(rname, None),
            )
            val_by_robot[rname] = v
            w = robot_weight.get(rname, 1.0)
            val_loss_weighted += w * v["loss"]
            wsum += w
        val_loss_weighted = val_loss_weighted / max(wsum, 1e-12)

        dt = time.time() - t0
        print(
            f"[Epoch {epoch:03d}] train loss={tr_loss:.6f} (dof={tr_dof:.6f}, root={tr_root:.6f}) "
            f"| val(avg) loss={val_loss_weighted:.6f} | {dt:.1f}s"
        )
        for rname, v in val_by_robot.items():
            print(f"    [Val {rname}] loss={v['loss']:.6f} dof={v['loss_dof']:.6f} root={v['loss_root']:.6f}")

        # save ckpt
        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "human_encoder": human_encoder.state_dict(),
            "robot_encoder": robot_encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optim": optim.state_dict(),
            "args": vars(args),
            "val_by_robot": val_by_robot,
            "val_avg": val_loss_weighted,
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))
        if val_loss_weighted < best_val:
            best_val = val_loss_weighted
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            print(f"  -> [Best] updated best_val(avg)={best_val:.6f}")

    print("Training done.")
    print(f"Best val(avg) loss: {best_val:.6f}")


# -----------------------------
# CLI
# -----------------------------

def build_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--robots_json", type=str, required=True, help="JSON list of robots (see docstring).")
    p.add_argument("--mix_mode", type=str, default="per_robot", choices=["per_robot", "roundrobin", "weighted"],
                   help="How to mix robots during training.")
    p.add_argument("--steps_per_epoch", type=int, default=0,
                   help="Only used for roundrobin/weighted. 0 means sum(len(loader_r)).")

    # Optional global fallbacks (if a robot entry does not provide its own input dirs / morph_npz)
    p.add_argument("--train_input_dir", type=str, default="", help="Fallback train input dir if missing in robot entry.")
    p.add_argument("--val_input_dir", type=str, default="", help="Fallback val input dir if missing in robot entry.")
    p.add_argument("--morph_npz", type=str, default="", help="Fallback morphology npz if missing in robot entry.")

    # data
    p.add_argument("--clip_len", type=int, default=64)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--suffix", type=str, default="_refine")
    p.add_argument("--fps_default", type=float, default=30.0)
    p.add_argument("--align_mode", type=str, default="min", choices=["min", "strict"])

    # features
    p.add_argument("--no_vel", action="store_true", help="Disable vel features (if your dataset supports vel).")
    p.add_argument("--vel_in_mps", action="store_true", help="Velocity uses meters/sec; else per-frame.")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--human_layers", type=int, default=4)
    p.add_argument("--robot_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--no_morph_film", action="store_true")
    p.add_argument("--limit_mode", type=str, default="tanh", choices=["tanh", "none"])

    # loss
    p.add_argument("--w_root", type=float, default=4.0)

    # train
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=1e-6, help="Cosine schedule floor (prevents lr -> 0).")
    p.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "none"])
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--log_every", type=int, default=200)

    # switches
    p.add_argument("--amp", action="store_true")
    p.add_argument("--freeze_robot_encoder", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
