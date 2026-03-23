import argparse
import json
import os
import numpy as np


END_SENTINEL = "__END_SITE__"


def parse_bvh_hierarchy_v2(bvh_path: str):
    """
    修复版 BVH HIERARCHY 解析：
    - 正确处理 End Site 的 { }，避免多 pop 导致 multiple roots
    - 提取 joints / parent / offset
    - 忽略 End Site 作为 joint，但会用 sentinel 保持括号栈一致
    """
    with open(bvh_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    joints = []
    parent = {}
    offset = {}

    stack = []  # joint name stack + END sentinel stack
    pending_joint = None
    pending_end_site = False

    def current_parent_joint():
        # 找到栈里最近的 joint（跳过 END sentinel）
        for x in reversed(stack):
            if x != END_SENTINEL:
                return x
        return None

    def add_joint(name, par):
        if name in parent:
            raise ValueError(f"Duplicated joint name: {name}")
        joints.append(name)
        parent[name] = par
        offset[name] = np.zeros(3, dtype=np.float32)

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line.upper().startswith("MOTION"):
            break

        toks = line.split()
        key = toks[0].upper()

        if key in ("ROOT", "JOINT"):
            name = toks[1]
            par = current_parent_joint()
            add_joint(name, par)
            pending_joint = name
            pending_end_site = False
            i += 1
            continue

        if key == "END" and len(toks) >= 2 and toks[1].upper() == "SITE":
            # End Site 不作为 joint，但后续会遇到 { OFFSET ... }
            pending_joint = None
            pending_end_site = True
            i += 1
            continue

        if key == "{":
            if pending_joint is not None:
                stack.append(pending_joint)
                pending_joint = None
            elif pending_end_site:
                stack.append(END_SENTINEL)
                pending_end_site = False
            # else: 可能是异常格式，忽略
            i += 1
            continue

        if key == "}":
            if stack:
                stack.pop()
            i += 1
            continue

        if key == "OFFSET":
            # OFFSET 只记录在 joint 上，End Site 的 OFFSET 忽略（你也可以扩展保留）
            if stack and stack[-1] != END_SENTINEL:
                cur = stack[-1]
                xyz = np.array([float(toks[1]), float(toks[2]), float(toks[3])], dtype=np.float32)
                offset[cur] = xyz
            i += 1
            continue

        # CHANNELS 等其它行忽略
        i += 1

    return joints, parent, offset


def build_arrays(joints, parent_dict, offset_dict, unit_scale: float):
    name2idx = {n: i for i, n in enumerate(joints)}
    n = len(joints)

    parent_idx = -np.ones((n,), dtype=np.int32)
    offset = np.zeros((n, 3), dtype=np.float32)

    for i, j in enumerate(joints):
        p = parent_dict[j]
        parent_idx[i] = name2idx[p] if p is not None else -1
        offset[i] = offset_dict[j] * unit_scale

    # bone_len：edge (joint -> parent) 的长度
    bone_len = np.linalg.norm(offset, axis=1).astype(np.float32)

    return name2idx, parent_idx, offset, bone_len


def check_tree(parent_idx):
    n = len(parent_idx)
    roots = np.where(parent_idx < 0)[0].tolist()
    if len(roots) != 1:
        return False, roots
    # 环检查（简单回溯）
    for i in range(n):
        seen = set()
        cur = i
        while parent_idx[cur] != -1:
            cur = int(parent_idx[cur])
            if cur in seen:
                return False, roots
            seen.add(cur)
    return True, roots


def lca_path_length(a_idx, b_idx, parent_idx, bone_len):
    """
    dist(a,b) = sum bone_len along edges on the path.
    bone_len[x] 表示 x -> parent[x] 的边长。
    """
    anc = {}
    dist_a = 0.0
    cur = a_idx
    anc[cur] = 0.0
    while parent_idx[cur] != -1:
        dist_a += float(bone_len[cur])
        cur = int(parent_idx[cur])
        anc[cur] = dist_a

    dist_b = 0.0
    cur = b_idx
    if cur in anc:
        return anc[cur] + dist_b
    while parent_idx[cur] != -1:
        dist_b += float(bone_len[cur])
        cur = int(parent_idx[cur])
        if cur in anc:
            return anc[cur] + dist_b

    return float("nan")


def default_target_names():
    return [
        "Hips", "Spine2",
        "LeftUpLeg", "LeftLeg", "LeftFootMod",
        "RightUpLeg", "RightLeg", "RightFootMod",
        "LeftArm", "LeftForeArm", "LeftHand",
        "RightArm", "RightForeArm", "RightHand",
    ]


def default_target_edges():
    return [
        ("Hips", "Spine2"),

        ("Hips", "LeftUpLeg"),
        ("LeftUpLeg", "LeftLeg"),
        ("LeftLeg", "LeftFootMod"),

        ("Hips", "RightUpLeg"),
        ("RightUpLeg", "RightLeg"),
        ("RightLeg", "RightFootMod"),

        ("Spine2", "LeftArm"),
        ("LeftArm", "LeftForeArm"),
        ("LeftForeArm", "LeftHand"),

        ("Spine2", "RightArm"),
        ("RightArm", "RightForeArm"),
        ("RightForeArm", "RightHand"),
    ]


def load_targets_from_json(json_path: str, table_key="ik_match_table1"):
    with open(json_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    table = cfg[table_key]
    names = []
    for _, v in table.items():
        h = v[0]
        if h not in names:
            names.append(h)
    return names


def apply_alias(name: str):
    # 你这份 BVH 里确实是 LeftFoot / RightFoot，没有 FootMod
    alias = {
        "LeftFootMod": "LeftFoot",
        "RightFootMod": "RightFoot",
    }
    return alias.get(name, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--unit_scale", type=float, default=1.0,
                    help="BVH单位到米：cm用0.01，米用1.0")
    ap.add_argument("--json", default=None, help="可选：bvh_to_g1.json，用于自动targets")
    ap.add_argument("--table_key", default="ik_match_table1")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    joints, parent_dict, offset_dict = parse_bvh_hierarchy_v2(args.bvh)
    name2idx, parent_idx, offset, bone_len = build_arrays(joints, parent_dict, offset_dict, args.unit_scale)

    ok, roots = check_tree(parent_idx)
    if not ok:
        print(f"[WARN] Tree root count != 1. roots={roots} (still something wrong in BVH parsing or file format)")
    else:
        print(f"[OK] Single root idx={roots[0]} name={joints[roots[0]]}")

    # full save
    full_npz = os.path.join(args.out_dir, "human_morphology_full_n1.npz")
    np.savez(
        full_npz,
        joints=np.array(joints, dtype=object),
        parent_idx=parent_idx,
        offset=offset,
        bone_len=bone_len,
    )

    # targets
    if args.json:
        targets = load_targets_from_json(args.json, args.table_key)
    else:
        targets = default_target_names()

    # apply alias for lookup
    targets_alias = [apply_alias(t) for t in targets]

    # keep only those exist
    targets_exist = []
    missing = []
    for t_raw, t_alias in zip(targets, targets_alias):
        if t_alias in name2idx:
            targets_exist.append(t_raw)  # 保留原名（比如 LeftFootMod），但内部查 alias
        else:
            missing.append(t_raw)

    # edges（保留原名），但计算时走 alias
    edges = default_target_edges()
    edges_exist = []
    lengths = []
    for a, b in edges:
        a2, b2 = apply_alias(a), apply_alias(b)
        if (a in targets_exist) and (b in targets_exist) and (a2 in name2idx) and (b2 in name2idx):
            d = lca_path_length(name2idx[a2], name2idx[b2], parent_idx, bone_len)
            edges_exist.append((a, b))
            lengths.append(d)

    target_npz = os.path.join(args.out_dir, "human_morphology_targets_n1.npz")
    np.savez(
        target_npz,
        target_names=np.array(targets_exist, dtype=object),
        target_edges=np.array(edges_exist, dtype=object),
        target_edge_lengths=np.asarray(lengths, dtype=np.float32),
        missing_targets=np.array(missing, dtype=object),
        alias_note=np.array(["LeftFootMod->LeftFoot, RightFootMod->RightFoot"], dtype=object),
    )

    print("[OK] Saved:")
    print("  full   :", full_npz)
    print("  targets:", target_npz)
    print(f"  #full_joints={len(joints)}  #targets_exist={len(targets_exist)}  #edges={len(edges_exist)}")
    if missing:
        print("  Missing targets:", missing)


if __name__ == "__main__":
    main()

"""
python net/morph_out/extract_morphology_bvh_full_and_targets.py \
  --bvh lafan1/aiming1_subject1.bvh \
  --json general_motion_retargeting/ik_configs/bvh_to_n1.json \
  --table_key ik_match_table1 \
  --out_dir net/morph_out \
  --unit_scale 0.01
"""