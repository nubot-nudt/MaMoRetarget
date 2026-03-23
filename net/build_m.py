import json
import re
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple


def load_robot_joint_names_from_mujoco_xml(xml_path: str) -> List[str]:
    """
    Returns joint names in MuJoCo XML traversal order.
    In your pipeline, dof_pos = qpos[7:], and MuJoCo's qpos joint order
    matches the XML joint order (after the free joint).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    joint_names = []
    for j in root.iter("joint"):
        name = j.attrib.get("name")
        if not name:
            continue
        # Commonly: all 29 are hinge joints with names. No free joint here.
        joint_names.append(name)

    return joint_names


def load_match_table(json_path: str, table_key: str = "ik_match_table1") -> Tuple[Dict[str, str], List[str]]:
    """
    From bvh_to_g1.json:
      ik_match_tableX[robot_link] = [human_joint, ..., ...]
    Returns:
      robot_link_to_human: dict robot_link -> human_joint
      human_targets_ordered: list of unique human joints in insertion order
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    table = data[table_key]  # dict: robot_link -> [human_joint, ...]
    robot_link_to_human = {robot_link: vals[0] for robot_link, vals in table.items()}

    human_targets_ordered = []
    for robot_link, vals in table.items():
        h = vals[0]
        if h not in human_targets_ordered:
            human_targets_ordered.append(h)

    return robot_link_to_human, human_targets_ordered


def guess_anchor_link_for_joint(joint_name: str, robot_link_to_human: Dict[str, str]) -> str:
    """
    Your match table keys are links like:
      left_hip_yaw_link, left_knee_link, left_ankle_roll_link, torso_link, ...
    But XML joints include pitch/roll/yaw variants like left_hip_pitch_joint, etc.

    This function maps each full robot joint to the nearest "anchor link" that exists in match table.
    """
    # 1) Try direct replace: xxx_joint -> xxx_link
    cand = joint_name.replace("_joint", "_link")
    if cand in robot_link_to_human:
        return cand

    # 2) If it's like <prefix>_(pitch|roll|yaw)_joint, try yaw_link / roll_link / pitch_link
    m = re.match(r"(.+?)_(pitch|roll|yaw)_joint$", joint_name)
    if m:
        prefix = m.group(1)
        for suf in ["yaw_link", "roll_link", "pitch_link"]:
            c = f"{prefix}_{suf}"
            if c in robot_link_to_human:
                return c

    # 3) Group heuristics for g1: map multi-DOF joints to the anchor link you actually set as target
    if joint_name.startswith("left_hip_"):
        return "left_hip_yaw_link"
    if joint_name.startswith("right_hip_"):
        return "right_hip_yaw_link"

    if joint_name.startswith("left_knee"):
        return "left_knee_link"
    if joint_name.startswith("right_knee"):
        return "right_knee_link"

    if joint_name.startswith("left_ankle_"):
        return "left_ankle_roll_link"
    if joint_name.startswith("right_ankle_"):
        return "right_ankle_roll_link"

    if joint_name.startswith("waist_"):
        return "torso_link"

    if joint_name.startswith("left_shoulder_"):
        return "left_shoulder_yaw_link"
    if joint_name.startswith("right_shoulder_"):
        return "right_shoulder_yaw_link"

    if joint_name.startswith("left_elbow"):
        return "left_elbow_link"
    if joint_name.startswith("right_elbow"):
        return "right_elbow_link"

    if joint_name.startswith("left_wrist_"):
        return "left_wrist_yaw_link"
    if joint_name.startswith("right_wrist_"):
        return "right_wrist_yaw_link"

    raise KeyError(f"Cannot map robot joint '{joint_name}' to any anchor link in match table.")


def build_M(
    robot_joint_names: List[str],
    human_target_names: List[str],
    robot_link_to_human: Dict[str, str],
    one_hot: bool = True,
) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Build M with rows = robot joints (full DOF joints),
             cols = human targets (the joints you set as IK targets).
    M[i, j] = 1 means robot joint i attends to human target j.

    Returns:
      M: (N_robot_joints, N_human_targets)
      robot_joint_to_human: mapping for debugging/inspection
    """
    human_index = {h: i for i, h in enumerate(human_target_names)}
    M = np.zeros((len(robot_joint_names), len(human_target_names)), dtype=np.float32)

    robot_joint_to_human = {}
    for i, rj in enumerate(robot_joint_names):
        anchor_link = guess_anchor_link_for_joint(rj, robot_link_to_human)
        hj = robot_link_to_human[anchor_link]
        if hj not in human_index:
            raise KeyError(f"Human joint '{hj}' from match table not in human_target_names list.")

        j = human_index[hj]
        robot_joint_to_human[rj] = hj

        if one_hot:
            M[i, j] = 1.0
        else:
            # You can make it soft here if you want (e.g., distribute weights).
            M[i, j] = 1.0

    return M, robot_joint_to_human


if __name__ == "__main__":
    json_path = "/home/nubot/GMR_old/general_motion_retargeting/ik_configs/bvh_to_g1.json"
    xml_path = "/home/nubot/GMR_old/assets/unitree_g1/g1_mocap_29dof.xml"

    robot_joint_names = load_robot_joint_names_from_mujoco_xml(xml_path)
    robot_link_to_human, human_target_names = load_match_table(json_path, table_key="ik_match_table1")

    M, robot_joint_to_human = build_M(
        robot_joint_names=robot_joint_names,
        human_target_names=human_target_names,
        robot_link_to_human=robot_link_to_human,
        one_hot=True,
    )

    print("robot_joint_names (rows):", len(robot_joint_names))
    print(robot_joint_names)
    print("human_target_names (cols):", len(human_target_names))
    print(human_target_names)
    print("M shape:", M.shape, "sum:", float(M.sum()))

    # Save
    np.savez("/home/nubot/GMR_old/net/retarget_M_g1.npz",
             M=M,
             robot_joint_names=np.array(robot_joint_names, dtype=object),
             human_target_names=np.array(human_target_names, dtype=object),
             robot_joint_to_human=json.dumps(robot_joint_to_human, ensure_ascii=False))

    # Also save readable CSV (header row + first column as joint name)
    import csv
    with open("/home/nubot/GMR_old/net/retarget_M_g1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["robot_joint"] + human_target_names)
        for i, rj in enumerate(robot_joint_names):
            w.writerow([rj] + M[i].astype(int).tolist())

    print("Saved: retarget_M_g1.npz, retarget_M_g1.csv")
