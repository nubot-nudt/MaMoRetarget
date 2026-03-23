#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export robot_spec.npz from a MuJoCo MJCF XML.

robot_spec is represented as:
- Node tokens: one token per DoF joint (hinge/slide), variable length Nr
- Tree/graph topology: parent_idx (Nr,), edge_index (2,E)
- Kinematic geometry at rest: joint pivot positions in base frame, offsets, lengths
- Joint constraints: axis, limits (q_min/q_max), rest_q(ref), joint_type

Notes:
- We treat the root body in <worldbody> as the robot base frame (pose set to identity),
  so all exported positions are in the base frame, NOT absolute world coordinates.
- <freejoint> is recognized as floating base and excluded from DoF joint tokens.
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET
from typing import Dict, Optional, Tuple

import numpy as np


# -----------------------------
# Basic parsers
# -----------------------------
def parse_floats(s: Optional[str], n: int, default=None) -> np.ndarray:
    if s is None:
        if default is None:
            raise ValueError("Missing attribute and no default provided.")
        return np.asarray(default, dtype=np.float32)
    vals = [float(x) for x in s.strip().split()]
    if len(vals) != n:
        raise ValueError(f"Expected {n} floats, got {len(vals)} from '{s}'")
    return np.asarray(vals, dtype=np.float32)


def quat_wxyz_to_R(q_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo body quat is w x y z."""
    q = np.asarray(q_wxyz, dtype=np.float32)
    q = q / max(np.linalg.norm(q), 1e-8)
    w, x, y, z = q

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)
    return R


def euler_to_R_xyz(e: np.ndarray, angle_unit: str) -> np.ndarray:
    """
    Fallback if some bodies use euler.
    In many MJCF robot models, body uses quat, so euler is rarely needed.
    We'll implement common convention: R = Rz @ Ry @ Rx.
    """
    x, y, z = [float(v) for v in e]
    if angle_unit != "radian":
        x, y, z = np.deg2rad([x, y, z])

    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy],
                   [0,  1, 0],
                   [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0],
                   [sz,  cz, 0],
                   [0,   0,  1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def body_local_transform(body_el: ET.Element, angle_unit: str) -> Tuple[np.ndarray, np.ndarray]:
    pos = parse_floats(body_el.get("pos"), 3, default=[0, 0, 0])

    if body_el.get("quat") is not None:
        q = parse_floats(body_el.get("quat"), 4, default=[1, 0, 0, 0])
        R = quat_wxyz_to_R(q)
    elif body_el.get("euler") is not None:
        e = parse_floats(body_el.get("euler"), 3, default=[0, 0, 0])
        R = euler_to_R_xyz(e, angle_unit=angle_unit)
    elif body_el.get("axisangle") is not None:
        aa = parse_floats(body_el.get("axisangle"), 4, default=[0, 0, 1, 0])
        axis = aa[:3]
        ang = float(aa[3])
        if angle_unit != "radian":
            ang = math.radians(ang)
        axis = axis / max(np.linalg.norm(axis), 1e-8)
        x, y, z = axis
        c, s = math.cos(ang), math.sin(ang)
        C = 1 - c
        R = np.array([
            [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ],
        ], dtype=np.float32)
    else:
        R = np.eye(3, dtype=np.float32)

    return R, pos


# -----------------------------
# MJCF default resolution (simple)
# -----------------------------
def build_default_joint_maps(mujoco_root: ET.Element) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
    """
    Very lightweight support for MJCF <default>:
    - global default: <default><joint .../></default>
    - class default : <default class="X"><joint .../></default>
    (nested defaults are partially supported)
    """
    global_joint: Dict[str, str] = {}
    class_joint: Dict[str, Dict[str, str]] = {}

    for d in mujoco_root.findall("default"):
        cls = d.get("class")
        j = d.find("joint")
        if j is not None:
            if cls is None:
                global_joint.update(j.attrib)
            else:
                class_joint.setdefault(cls, {}).update(j.attrib)

        # nested <default class="...">
        for sub in d.findall("default"):
            sub_cls = sub.get("class")
            j2 = sub.find("joint")
            if j2 is not None and sub_cls is not None:
                class_joint.setdefault(sub_cls, {}).update(j2.attrib)

    return global_joint, class_joint


def get_joint_attr(joint_el: ET.Element, key: str, eff_class: Optional[str],
                   global_joint: Dict[str, str], class_joint: Dict[str, Dict[str, str]]) -> Optional[str]:
    if joint_el.get(key) is not None:
        return joint_el.get(key)
    if eff_class is not None and eff_class in class_joint and key in class_joint[eff_class]:
        return class_joint[eff_class][key]
    if key in global_joint:
        return global_joint[key]
    return None


# -----------------------------
# Main export
# -----------------------------
def export_robot_spec(xml_path: str) -> Dict[str, np.ndarray]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    angle_unit = compiler.get("angle", "degree") if compiler is not None else "degree"

    global_joint, class_joint = build_default_joint_maps(root)

    worldbodies = root.findall("worldbody")
    if not worldbodies:
        raise ValueError("No <worldbody> found in XML.")

    # pick first non-empty worldbody
    wb = None
    for w in worldbodies:
        if w.find("body") is not None:
            wb = w
            break
    if wb is None:
        wb = worldbodies[0]

    base_body = wb.find("body")
    if base_body is None:
        raise ValueError("<worldbody> has no <body>.")

    base_body_name = base_body.get("name", "base")
    base_pos_worldbody = parse_floats(base_body.get("pos"), 3, default=[0, 0, 0])
    base_has_freejoint = (base_body.find("freejoint") is not None) or any(
        (j.get("type") == "free" for j in base_body.findall("joint"))
    )

    # storage
    joint_names = []
    body_names = []
    parent_idx = []
    axis_local = []
    joint_type = []
    q_min, q_max, limited = [], [], []
    rest_q = []

    joint_pos_base = []
    parent_pos_base = []
    offset_base = []

    joint_pos_list = []  # for parent lookup by index

    def traverse_body(body_el: ET.Element,
                      parent_R: np.ndarray, parent_p: np.ndarray,
                      parent_joint: int,
                      inherited_childclass: Optional[str]) -> None:
        # childclass inheritance
        this_childclass = body_el.get("childclass", inherited_childclass)

        # body transform in base frame
        R_rel, p_rel = body_local_transform(body_el, angle_unit=angle_unit)
        p = parent_p + parent_R @ p_rel
        R = parent_R @ R_rel

        # multiple joints in same body: chain them in XML order
        last_joint = parent_joint

        for j_el in body_el.findall("joint"):
            eff_class = j_el.get("class", this_childclass)

            jtype = get_joint_attr(j_el, "type", eff_class, global_joint, class_joint) or "hinge"
            if jtype in ("free", "ball"):
                # freejoint/ball: not part of 1D dof list here
                continue

            jname = get_joint_attr(j_el, "name", eff_class, global_joint, class_joint)
            if jname is None:
                continue

            jpos = parse_floats(get_joint_attr(j_el, "pos", eff_class, global_joint, class_joint),
                                3, default=[0, 0, 0])
            jaxis = parse_floats(get_joint_attr(j_el, "axis", eff_class, global_joint, class_joint),
                                 3, default=[0, 0, 1])
            n = np.linalg.norm(jaxis)
            if n > 1e-8:
                jaxis = jaxis / n

            rng = get_joint_attr(j_el, "range", eff_class, global_joint, class_joint)
            lim_attr = get_joint_attr(j_el, "limited", eff_class, global_joint, class_joint)
            lim = None
            if lim_attr is not None:
                lim = (lim_attr.lower() == "true" or lim_attr == "1")

            if rng is not None:
                rv = [float(x) for x in rng.strip().split()]
                if len(rv) == 2:
                    qmn, qmx = rv
                else:
                    qmn, qmx = -math.pi, math.pi
                lim = True if lim is None else lim
            else:
                # fallback (still useful numerically, and limited=0 tells the network it's “soft”)
                qmn, qmx = -math.pi, math.pi
                if lim is None:
                    lim = False

            ref = get_joint_attr(j_el, "ref", eff_class, global_joint, class_joint)
            refv = float(ref) if ref is not None else 0.0

            # pivot position in base frame
            jp = p + R @ jpos

            # parent pivot position in base frame
            if last_joint >= 0:
                pp = joint_pos_list[last_joint]
            else:
                pp = np.zeros(3, dtype=np.float32)

            off = jp - pp

            idx = len(joint_names)
            joint_names.append(jname)
            body_names.append(body_el.get("name", ""))

            parent_idx.append(last_joint)
            axis_local.append(jaxis.astype(np.float32))
            joint_type.append({"hinge": 0, "slide": 1}.get(jtype, 9))

            q_min.append(qmn)
            q_max.append(qmx)
            limited.append(1 if lim else 0)
            rest_q.append(refv)

            joint_pos_list.append(jp.astype(np.float32))
            joint_pos_base.append(jp.astype(np.float32))
            parent_pos_base.append(pp.astype(np.float32))
            offset_base.append(off.astype(np.float32))

            last_joint = idx  # chain within the same body

        # traverse children
        for child in body_el.findall("body"):
            traverse_body(child, R, p, last_joint, this_childclass)

    # Base frame is identity at base_body (ignore its worldbody pos for invariance)
    traverse_body(
        base_body,
        parent_R=np.eye(3, dtype=np.float32),
        parent_p=np.zeros(3, dtype=np.float32),
        parent_joint=-1,
        inherited_childclass=base_body.get("childclass", None),
    )

    Nr = len(joint_names)
    if Nr == 0:
        raise ValueError("No hinge/slide joints found. Check XML structure.")

    joint_names_arr = np.array(joint_names, dtype=object)
    body_names_arr = np.array(body_names, dtype=object)
    parent_idx_arr = np.asarray(parent_idx, dtype=np.int32)

    axis_local_arr = np.stack(axis_local, axis=0).astype(np.float32)
    joint_type_arr = np.asarray(joint_type, dtype=np.int32)

    q_min_arr = np.asarray(q_min, dtype=np.float32)
    q_max_arr = np.asarray(q_max, dtype=np.float32)
    limited_arr = np.asarray(limited, dtype=np.uint8)
    rest_q_arr = np.asarray(rest_q, dtype=np.float32)

    joint_pos_base_arr = np.stack(joint_pos_base, axis=0).astype(np.float32)
    parent_pos_base_arr = np.stack(parent_pos_base, axis=0).astype(np.float32)
    offset_base_arr = np.stack(offset_base, axis=0).astype(np.float32)
    offset_len_arr = np.linalg.norm(offset_base_arr, axis=1).astype(np.float32)

    # edge list (tree edges)
    edges = [(p, i) for i, p in enumerate(parent_idx) if p >= 0]
    edge_index = np.array(edges, dtype=np.int32).T if edges else np.zeros((2, 0), dtype=np.int32)

    spec = dict(
        base_body_name=np.array(base_body_name, dtype=object),
        base_pos_worldbody=base_pos_worldbody.astype(np.float32),
        base_has_freejoint=np.array([1 if base_has_freejoint else 0], dtype=np.uint8),

        joint_names=joint_names_arr,
        body_names=body_names_arr,
        parent_idx=parent_idx_arr,
        edge_index=edge_index,

        axis_local=axis_local_arr,      # (Nr,3)
        joint_type=joint_type_arr,      # (Nr,) hinge=0, slide=1, other=9
        q_min=q_min_arr,                # (Nr,)
        q_max=q_max_arr,                # (Nr,)
        limited=limited_arr,            # (Nr,) 0/1
        rest_q=rest_q_arr,              # (Nr,)

        joint_pos_base=joint_pos_base_arr,   # (Nr,3) pivot positions in base frame
        parent_pos_base=parent_pos_base_arr, # (Nr,3)
        offset_base=offset_base_arr,         # (Nr,3) = joint_pos - parent_pos
        offset_len=offset_len_arr,           # (Nr,)
    )
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", type=str, required=True, help="Path to MJCF xml (MuJoCo)")
    ap.add_argument("--out", type=str, default="robot_spec.npz", help="Output npz path")
    ap.add_argument("--print", action="store_true", help="Print a short summary")
    args = ap.parse_args()

    spec = export_robot_spec(args.xml)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **spec)

    if args.print:
        Nr = spec["joint_names"].shape[0]
        roots = np.where(spec["parent_idx"] < 0)[0].tolist()
        print(f"[robot_spec] saved: {args.out}")
        print(f"  Nr joints: {Nr}")
        print(f"  roots (parent=-1): {roots}")
        print(f"  base_body_name: {spec['base_body_name'].item()}")
        print(f"  base_has_freejoint: {int(spec['base_has_freejoint'][0])}")
        print("  first 10 joints:", [spec["joint_names"][i] for i in range(min(10, Nr))])
        print("  offset_len stats:",
              float(spec["offset_len"].min()),
              float(spec["offset_len"].mean()),
              float(spec["offset_len"].max()))

if __name__ == "__main__":
    main()


"""
python export_robot_spec_from_mujoco_xml.py \
  --xml xml/engineai_pm01/pm_v2_full.xml \
  --out robot_spec_pm01.npz \
  --print
"""

"""
获取robot_spec_g1.npz,用于robot_spec_encoder。
"""