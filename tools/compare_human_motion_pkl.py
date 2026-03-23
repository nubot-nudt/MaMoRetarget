import argparse
import pickle
import numpy as np

def load_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def to_np(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    # torch tensor etc.
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def compare_array(name, a, b, atol=1e-6, rtol=1e-6):
    a = to_np(a); b = to_np(b)
    if a is None or b is None:
        print(f"[{name}] one is None? a={a is None}, b={b is None}")
        return
    if a.shape != b.shape:
        print(f"[{name}] shape mismatch: {a.shape} vs {b.shape}")
        return
    diff = np.abs(a - b)
    mx = diff.max() if diff.size else 0.0
    mean = diff.mean() if diff.size else 0.0
    ok = np.allclose(a, b, atol=atol, rtol=rtol)
    print(f"[{name}] allclose={ok} | max_abs_diff={mx:.3e} mean_abs_diff={mean:.3e}")

def compare_quat(name, qa, qb, atol=1e-6, rtol=1e-6):
    """Handle quaternion sign ambiguity: q and -q represent same rotation."""
    qa = to_np(qa); qb = to_np(qb)
    if qa is None or qb is None:
        print(f"[{name}] one is None? a={qa is None}, b={qb is None}")
        return
    if qa.shape != qb.shape:
        print(f"[{name}] shape mismatch: {qa.shape} vs {qb.shape}")
        return
    # choose min(|qa-qb|, |qa+qb|)
    diff1 = np.abs(qa - qb)
    diff2 = np.abs(qa + qb)
    diff = np.minimum(diff1, diff2)
    mx = diff.max() if diff.size else 0.0
    mean = diff.mean() if diff.size else 0.0
    ok = np.all(diff <= (atol + rtol * np.abs(qa)))
    print(f"[{name}] (quat sign-invariant) ok={ok} | max_abs_diff={mx:.3e} mean_abs_diff={mean:.3e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="pkl file A")
    ap.add_argument("--b", required=True, help="pkl file B")
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--rtol", type=float, default=1e-6)
    args = ap.parse_args()

    da = load_pkl(args.a)
    db = load_pkl(args.b)

    ha = da.get("human_motion", {})
    hb = db.get("human_motion", {})

    # 1) link list
    la = ha.get("link_body_list", None)
    lb = hb.get("link_body_list", None)


    name2idx_a = {n:i for i,n in enumerate(la)}
    perm = [name2idx_a[n] for n in lb]   # 把A重排成B的顺序

    a_local = to_np(ha["local_body_pos"])[:, perm, :]
    b_local = to_np(hb["local_body_pos"])

    compare_array("local_body_pos (A reordered to B)", a_local, b_local, atol=1e-6, rtol=1e-6)


    same_list = (la == lb)
    print(f"[link_body_list] same={same_list} | len_a={len(la) if la is not None else None} len_b={len(lb) if lb is not None else None}")
    if not same_list and la is not None and lb is not None:
        # show first mismatch
        n = min(len(la), len(lb))
        for i in range(n):
            if la[i] != lb[i]:
                print(f"  first mismatch @ {i}: {la[i]} vs {lb[i]}")
                break

    # 2) fps + shapes
    print(f"[fps] a={ha.get('fps', None)} b={hb.get('fps', None)}")
    for k in ["root_pos", "local_body_pos"]:
        xa, xb = ha.get(k, None), hb.get(k, None)
        if xa is not None and xb is not None:
            xa, xb = to_np(xa), to_np(xb)
            print(f"[{k}] shape a={xa.shape} b={xb.shape}")
        else:
            print(f"[{k}] exists a={xa is not None} b={xb is not None}")

    # 3) numeric comparison
    compare_array("root_pos", ha.get("root_pos", None), hb.get("root_pos", None), args.atol, args.rtol)
    compare_quat("root_rot", ha.get("root_rot", None), hb.get("root_rot", None), args.atol, args.rtol)
    compare_array("local_body_pos", ha.get("local_body_pos", None), hb.get("local_body_pos", None), args.atol, args.rtol)


import pickle, numpy as np

def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)

def allclose(a,b, name):
    a = np.asarray(a); b = np.asarray(b)
    print(name, "shape", a.shape, b.shape,
          "allclose", np.allclose(a,b, atol=1e-6, rtol=1e-6),
          "max", np.max(np.abs(a-b)))

A = load("retargeting_result/g1_gmr/aiming1_subject1_c.pkl")["robot_motion"]
B = load("net/dataset/g1/train/input/aiming1_subject1.pkl")["robot_motion"]

allclose(A["root_pos"], B["root_pos"], "root_pos")
allclose(A["root_rot"], B["root_rot"], "root_rot")
allclose(A["dof_pos"],  B["dof_pos"],  "dof_pos")

print("A has dof_names?", "dof_names" in A, "B has dof_names?", "dof_names" in B)


if __name__ == "__main__":
    main()


"""
python tools/compare_human_motion_pkl.py --a retargeting_result/g1_gmr/aiming1_subject1_c.pkl --b net/dataset/g1/train/input/aiming1_subject1.pkl

"""