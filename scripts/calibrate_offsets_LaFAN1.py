# scripts/calibrate_offsets_LaFAN1.py
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R

import mujoco as mj
import mink
# 不再 import FrameType，做动态兼容

from general_motion_retargeting.utils.lafan1 import load_lafan1_file


def quat_norm_wxyz(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    return q if n == 0 else q / n


def scale_human_data_like_runtime(human_data, human_root_name, human_scale_table):
    root_pos, root_quat = human_data[human_root_name]
    scale_root = float(human_scale_table.get(human_root_name, 1.0))
    scaled_root_pos = scale_root * root_pos

    scaled = {human_root_name: (scaled_root_pos, root_quat)}
    for name, (pos, quat) in human_data.items():
        if name == human_root_name:
            continue
        s = human_scale_table.get(name, None)
        if s is None:
            scaled[name] = (pos, quat)
        else:
            scaled_pos = (pos - root_pos) * float(s) + scaled_root_pos
            scaled[name] = (scaled_pos, quat)
    return scaled


def get_body_tf(config, body_name):
    """
    兼容不同 mink 版本获取 body 世界位姿：
    1) 试字符串 'body'
    2) 试可能存在的枚举 FrameType.BODY（在多个命名空间里找）
    3) 兜底尝试无 frame_type 形参的旧签名
    """
    # 1) 字符串帧类型（很多版本支持）
    for key in ("body", "BODY", "Body"):
        try:
            return config.get_transform_frame_to_world(body_name, key)
        except TypeError:
            # 形参不匹配，继续尝试别的
            pass
        except Exception:
            # 其他错误（比如 body_name 不存在），直接抛给上层
            raise

    # 2) 不同位置查找枚举
    candidates = (
        "mink", "mink.configuration", "mink.core.enums", "mink.enums", "mink.types"
    )
    for modname in candidates:
        try:
            mod = __import__(modname, fromlist=["FrameType"])
            FT = getattr(mod, "FrameType", None)
            if FT is not None:
                try:
                    return config.get_transform_frame_to_world(body_name, FT.BODY)
                except TypeError:
                    pass
        except Exception:
            pass

    # 3) 兜底：老版本可能不需要 frame_type
    try:
        return config.get_transform_frame_to_world(body_name)
    except Exception as e:
        raise e  # 彻底不支持，抛出让外层打印


def calibrate(bvh_file, robot_xml, ik_in, ik_out, t_ref=0):
    out = load_lafan1_file(bvh_file)
    if isinstance(out, tuple) and len(out) == 2:
        frames, actual_human_height = out
    else:
        frames = out
        actual_human_height = None

    frame0 = frames[t_ref]
    human0 = {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in frame0.items()}

    ik_cfg = json.loads(Path(ik_in).read_text())
    human_root = ik_cfg["human_root_name"]
    scale_table = dict(ik_cfg.get("human_scale_table", {}))

    if actual_human_height is None:
        actual_human_height = float(ik_cfg.get("human_height_assumption", 1.8))
    ratio = float(actual_human_height) / float(ik_cfg.get("human_height_assumption", 1.8))
    for k in list(scale_table.keys()):
        scale_table[k] = float(scale_table[k]) * ratio

    if human_root not in human0:
        raise KeyError(f"human_root_name '{human_root}' 不在 BVH 骨架里，可用键例子: {list(human0.keys())[:10]} ...")

    human0_scaled = scale_human_data_like_runtime(human0, human_root, scale_table)

    model = mj.MjModel.from_xml_path(str(robot_xml))
    config = mink.Configuration(model)  # 默认 qpos=0

    def as_dict(tbl):
        return {rb: spec for rb, spec in tbl.items() if isinstance(spec, list) and len(spec) >= 5}

    tbl1 = as_dict(ik_cfg.get("ik_match_table1", {}))
    tbl2 = as_dict(ik_cfg.get("ik_match_table2", {}))
    pairs = dict(tbl1)
    for k, v in tbl2.items():
        pairs.setdefault(k, v)  # 表1优先，表2补充

    for rb, spec in pairs.items():
        hb, pos_w, ori_w, pos_off_old, rot_off_old = spec[:5]

        if hb not in human0_scaled:
            print(f"[skip] BVH 不含骨骼 '{hb}'，跳过 '{rb}' 的标定")
            continue

        ph, qh = human0_scaled[hb]
        Rh = R.from_quat(quat_norm_wxyz(qh), scalar_first=True)

        try:
            Tf = get_body_tf(config, rb)
        except Exception as e:
            print(f"[skip] 取 '{rb}' 世界位姿失败：{e}")
            continue

        try:
            qr = Tf.rotation.as_quat(scalar_first=True); pr = np.asarray(Tf.translation)
        except AttributeError:
            qr = Tf.rotation().as_quat(scalar_first=True); pr = np.asarray(Tf.translation())
        Rr = R.from_quat(quat_norm_wxyz(qr), scalar_first=True)

        Roff = Rh.inv() * Rr
        poff = Rh.inv().apply(pr - ph)

        spec[3] = [float(f"{v:.6f}") for v in poff]
        spec[4] = [float(f"{v:.6f}") for v in quat_norm_wxyz(Roff.as_quat(scalar_first=True))]

    Path(ik_out).write_text(json.dumps(ik_cfg, indent=2))
    print(f"[OK] wrote calibrated IK config to: {ik_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bvh_file", required=True)
    p.add_argument("--robot_xml", required=True)
    p.add_argument("--ik_in", required=True)
    p.add_argument("--ik_out", required=True)
    p.add_argument("--t_ref", type=int, default=0)
    args = p.parse_args()
    calibrate(args.bvh_file, args.robot_xml, args.ik_in, args.ik_out, t_ref=args.t_ref)
