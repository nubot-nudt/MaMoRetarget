import argparse
import numpy as np
from collections import defaultdict, deque

def load_npz(path):
    return np.load(path, allow_pickle=True)

def check_tree(parent_idx):
    n = len(parent_idx)
    roots = np.where(parent_idx < 0)[0].tolist()
    if len(roots) == 0:
        return False, f"[ERR] No root found (no parent_idx < 0)."
    if len(roots) > 1:
        # BVH 通常只有一个 ROOT；多个 root 可能意味着解析异常或 BVH 特殊结构
        msg = f"[WARN] Multiple roots found: {roots}"
    else:
        msg = f"[OK] Single root: {roots[0]}"

    # 检查父索引范围 & self-loop
    for i, p in enumerate(parent_idx):
        if p >= n:
            return False, f"[ERR] parent_idx[{i}]={p} out of range (n={n})"
        if p == i:
            return False, f"[ERR] self-loop at node {i}"

    # 检查环：沿父链回溯
    for i in range(n):
        seen = set()
        cur = i
        while parent_idx[cur] != -1:
            cur = int(parent_idx[cur])
            if cur in seen:
                return False, f"[ERR] cycle detected starting at node {i}, loop at {cur}"
            seen.add(cur)

    return True, msg

def print_tree(joints, parent_idx, offset, bone_len, max_nodes=200):
    n = len(joints)
    children = defaultdict(list)
    roots = []
    for i, p in enumerate(parent_idx):
        if p < 0:
            roots.append(i)
        else:
            children[int(p)].append(i)

    def node_line(i, depth):
        name = str(joints[i])
        p = int(parent_idx[i])
        off = offset[i]
        bl = float(bone_len[i])
        return "  " * depth + f"- [{i}] {name}  parent={p}  offset={off.tolist()}  bone_len={bl:.5f}"

    # BFS/DFS 打印（多 root 也支持）
    printed = 0
    for r in roots:
        stack = [(r, 0)]
        while stack:
            i, d = stack.pop()
            print(node_line(i, d))
            printed += 1
            if printed >= max_nodes:
                print(f"... (stop at {max_nodes} nodes; use --max_nodes to print more)")
                return
            # 子节点按 index 排序，保证稳定输出
            for ch in sorted(children.get(i, []), reverse=True):
                stack.append((ch, d + 1))

def print_targets(target_names, target_edges, target_edge_lengths, missing_targets):
    print("\n[Target-only topology]")
    print("Targets exist (#={}):".format(len(target_names)))
    print("  " + ", ".join([str(x) for x in target_names.tolist()]))

    if missing_targets is not None and len(missing_targets) > 0:
        print("\n[WARN] Missing targets in this BVH (not found in hierarchy):")
        print("  " + ", ".join([str(x) for x in missing_targets.tolist()]))

    print("\nTarget edges (#={}):".format(len(target_edges)))
    bad = 0
    for (e, L) in zip(target_edges.tolist(), target_edge_lengths.tolist()):
        a, b = e
        if not np.isfinite(L) or L <= 0:
            bad += 1
            tag = "  [ERR?]"
        else:
            tag = ""
        print(f"  {a:12s} -> {b:12s}   path_len={float(L):.5f}{tag}")
    if bad > 0:
        print(f"\n[WARN] {bad} edges have non-finite or non-positive length. Check BVH joint names / parsing / unit_scale.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_npz", required=True, help="human_morphology_full.npz")
    ap.add_argument("--target_npz", required=True, help="human_morphology_targets.npz")
    ap.add_argument("--max_nodes", type=int, default=200, help="Max nodes to print from full tree")
    args = ap.parse_args()

    # -------- full --------
    full = load_npz(args.full_npz)
    joints = full["joints"]
    parent_idx = full["parent_idx"].astype(np.int32)
    offset = full["offset"].astype(np.float32)
    bone_len = full["bone_len"].astype(np.float32)

    print("[Full BVH topology]")
    print(f"  #joints = {len(joints)}")
    ok, msg = check_tree(parent_idx)
    print(" ", msg)
    if not ok:
        print("[STOP] Tree check failed.")
        return

    # 一些统计
    nonzero = bone_len[bone_len > 1e-8]
    if len(nonzero) > 0:
        print(f"  bone_len (nonzero) stats: min={nonzero.min():.5f}, mean={nonzero.mean():.5f}, max={nonzero.max():.5f}")
    else:
        print("  bone_len all ~0 (unexpected)")

    print("\n[Full tree print]")
    print_tree(joints, parent_idx, offset, bone_len, max_nodes=args.max_nodes)

    # -------- targets --------
    tgt = load_npz(args.target_npz)
    target_names = tgt["target_names"]
    target_edges = tgt["target_edges"]
    target_edge_lengths = tgt["target_edge_lengths"].astype(np.float32)
    missing_targets = tgt["missing_targets"] if "missing_targets" in tgt.files else None

    print_targets(target_names, target_edges, target_edge_lengths, missing_targets)

if __name__ == "__main__":
    main()


"""
python net/morph_out/print_morphology_npz.py \
  --full_npz  net/morph_out/human_morphology_full_t1.npz \
  --target_npz net/morph_out/human_morphology_targets_t1.npz \
  --max_nodes 300
"""