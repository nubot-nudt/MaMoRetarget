import re, os, ntpath
import numpy as np
from . import utils

channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}


class Anim(object):
    """
    A very basic animation object
    """
    def __init__(self, quats, pos, offsets, parents, bones):
        """
        :param quats: local quaternions tensor
        :param pos: local positions tensor
        :param offsets: local joint offsets
        :param parents: bone hierarchy
        :param bones: bone names
        """
        self.quats = quats
        self.pos = pos
        self.offsets = offsets
        self.parents = parents
        self.bones = bones


"""
Anim.quats(这里传入的是 rotations);
形状;(T, N, 4),T 为帧数,N 为关节数
每一帧、每个关节的局部四元数旋转(相对于父关节)

Anim.pos(positions);
形状;(T, N, 3)
每一帧、每个关节的“局部位置”;

对于 根关节 Hips;是真正随时间变化的平移轨迹(来自 MOTION 部分的前三个数)

对于其他关节;是静态的 offset(骨骼长度 / 父子关节间的相对位置),不随帧变化

Anim.offsets;形状 (N, 3),每个关节相对父关节的 OFFSET

Anim.parents;形状 (N,),整型数组,parents[i] = 该关节的父关节索引；根结点父亲为 -1

Anim.bones(names);长度 N 的字符串列表,关节名
"""

def read_bvh(filename, start=None, end=None, order=None):
    """
    Reads a BVH file and extracts animation information.

    :param filename: BVh filename
    :param start: start frame
    :param end: end frame
    :param order: order of euler rotations
    :return: A simple Anim object conatining the extracted information.
    """

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = np.array([]).reshape((0, 4))
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the  file, line by line
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    rotations = utils.euler_to_quat(np.radians(rotations), order=order)
    rotations = utils.remove_quat_discontinuities(rotations)

    return Anim(rotations, positions, offsets, parents, names)

def read_bvh_ourdata(filename, start=None, end=None, order=None, keep_all_joint_positions: bool=False):
    """
    读取 BVH,稳健支持;
      - 根6/其余3(root 有平移,其它只有旋转)
      - 全6(每个关节都有平移+旋转)
    参数;
      start, end: 选帧的半开区间 [start, end),None 表示不裁剪
      order: 若提供则覆盖文件中的欧拉旋转顺序(如 "ZYX")
      keep_all_joint_positions: 是否保留每个关节的 position(默认 False;仅 root)
    返回;
      Anim(rotations_quat, positions, offsets, parents, names)
    """
    channelmap = {
    "Xrotation": "X", "Yrotation": "Y", "Zrotation": "Z",
    "Xposition": "Xp", "Yposition": "Yp", "Zposition": "Zp"
    }
    from typing import List, Tuple, Optional

    names: List[str] = []
    offsets = np.zeros((0, 3), dtype=np.float64)
    parents = np.array([], dtype=np.int32)
    joint_channels: List[List[str]] = []   # 每个关节的 CHANNELS 列表
    layout: List[Tuple[int,int]] = []      # 每个关节在帧向量中的 [start_idx, end_idx)

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f]

    # ---- 解析层级,建立每个关节的通道列表(以及帧向量布局) ----
    i = 0
    while i < len(lines) and lines[i].strip() != "HIERARCHY":
        i += 1
    assert i < len(lines), "No HIERARCHY section"
    i += 1

    # 栈式解析
    active = -1
    end_site = False
    channel_cursor = 0  # 累计帧向量的指针

    while i < len(lines):
        line = lines[i].strip()
        if line == "MOTION":
            break
        if m := re.match(r"ROOT\s+(\S+)", line):
            names.append(m.group(1))
            offsets = np.vstack([offsets, np.zeros((1,3))])
            parents = np.append(parents, active)
            joint_channels.append([])
            layout.append((channel_cursor, channel_cursor))  # 先占位
            active = len(parents) - 1
            i += 1
            continue
        if m := re.match(r"JOINT\s+(\S+)", line):
            names.append(m.group(1))
            offsets = np.vstack([offsets, np.zeros((1,3))])
            parents = np.append(parents, active)
            joint_channels.append([])
            layout.append((channel_cursor, channel_cursor))
            active = len(parents) - 1
            i += 1
            continue
        if line == "{":
            i += 1; continue
        if line == "}":
            # 关闭当前关节
            active = parents[active] if active != -1 else -1
            i += 1; continue
        if m := re.match(r"OFFSET\s+([-\d\.eE]+)\s+([-\d\.eE]+)\s+([-\d\.eE]+)", line):
            offsets[active] = np.array(list(map(float, m.groups())))
            i += 1; continue
        if m := re.match(r"CHANNELS\s+(\d+)\s+(.*)", line):
            n = int(m.group(1))
            parts = m.group(2).strip().split()[:n]
            joint_channels[active] = parts
            # 为该关节记录它在帧向量的切片
            layout[active] = (channel_cursor, channel_cursor + n)
            channel_cursor += n
            # 若 order 未指定,从第一处旋转通道推断
            if order is None:
                rot = [p for p in parts if p.endswith("rotation")]
                if len(rot) == 3 and all(p in channelmap for p in rot):
                    order = "".join(channelmap[p] for p in rot).replace("p","")
            i += 1; continue
        if line.startswith("End Site"):
            # 跳过 End Site 小块;End Site { OFFSET ... }
            # 不参与通道和帧数据
            i += 1  # "{"
            i += 1  # "OFFSET ..."
            i += 1  # "}"
            continue
        i += 1

    assert lines[i].strip() == "MOTION"; i += 1

    # ---- MOTION 头 ----
    m = re.match(r"\s*Frames:\s+(\d+)", lines[i]); assert m
    frames_total = int(m.group(1)); i += 1
    m = re.match(r"\s*Frame Time:\s+([-\d\.eE]+)", lines[i]); assert m
    frame_time = float(m.group(1)); i += 1

    # ---- 选帧范围 ----
    if start is None: start = 0
    if end   is None: end = frames_total
    assert 0 <= start <= end <= frames_total
    fnum = end - start

    N = len(names)
    positions = np.repeat(offsets[np.newaxis, :, :], fnum, axis=0)  # 先填 offsets
    rotations_euler = np.zeros((fnum, N, 3), dtype=np.float64)

    # ---- 帧数据逐关节按 layout 解析 ----
    # 注意;BVH 的通道是按层级先序遍历依次排布的,layout 就是每关节在帧向量中的切片
    for fi_src in range(frames_total):
        row = lines[i + fi_src].strip()
        if not (start <= fi_src < end):
            continue
        fi = fi_src - start
        data = np.fromstring(row, sep=' ')
        # 基本健壮性检查
        assert data.size == channel_cursor, f"Frame {fi_src}: expected {channel_cursor} values, got {data.size}"

        for j in range(N):
            s, e = layout[j]
            chs = joint_channels[j]
            seg = data[s:e]

            # 提取 position/rotation(若不存在相应通道,则保持默认值)d
            if "Xposition" in chs and "Yposition" in chs and "Zposition" in chs:
                x = seg[chs.index("Xposition")]
                y = seg[chs.index("Yposition")]
                z = seg[chs.index("Zposition")]
                positions[fi, j] = [x, y, z]

            # rotation(度 -> 弧度,按文件通道顺序映射到 XYZ)
            # 这里把三个旋转通道按它们各自的字母落到 rotations_euler 的相应轴里
            for k, ch in enumerate(chs):
                if ch.endswith("rotation"):
                    axis = ch[0]  # 'X'/'Y'/'Z'
                    val_deg = seg[k]
                    if axis == "X": rotations_euler[fi, j, 0] = np.deg2rad(val_deg)
                    if axis == "Y": rotations_euler[fi, j, 1] = np.deg2rad(val_deg)
                    if axis == "Z": rotations_euler[fi, j, 2] = np.deg2rad(val_deg)

    # ---- 若仅保留 root 平移;把非 root 的 position 还原为 offsets(或置零) ----
    if not keep_all_joint_positions:
        positions[:, 1:, :] = offsets[1:][np.newaxis, :, :]

    # ---- 旋转欧拉 -> 四元数,并移除不连续 ----
    if order is None:
        order = "ZYX"  # 兜底(你的两个文件都是 ZYX)
    rotations = utils.euler_to_quat(rotations_euler, order=order)
    rotations = utils.remove_quat_discontinuities(rotations)

    return Anim(rotations, positions, offsets, parents, names)


def get_lafan1_set(bvh_path, actors, window=50, offset=20):
    """
    Extract the same test set as in the article, given the location of the BVH files.

    :param bvh_path: Path to the dataset BVH files
    :param list: actor prefixes to use in set
    :param window: width  of the sliding windows (in timesteps)
    :param offset: offset between windows (in timesteps)
    :return: tuple:
        X: local positions
        Q: local quaternions
        parents: list of parent indices defining the bone hierarchy
        contacts_l: binary tensor of left-foot contacts of shape (Batchsize, Timesteps, 2)
        contacts_r: binary tensor of right-foot contacts of shape (Batchsize, Timesteps, 2)
    """
    npast = 10
    subjects = []
    seq_names = []
    X = []
    Q = []
    contacts_l = []
    contacts_r = []

    # Extract
    bvh_files = os.listdir(bvh_path)

    for file in bvh_files:
        if file.endswith('.bvh'):
            seq_name, subject = ntpath.basename(file[:-4]).split('_')

            if subject in actors:
                print('Processing file {}'.format(file))
                seq_path = os.path.join(bvh_path, file)
                anim = read_bvh(seq_path)

                # Sliding windows
                i = 0
                while i+window < anim.pos.shape[0]:
                    q, x = utils.quat_fk(anim.quats[i: i+window], anim.pos[i: i+window], anim.parents)
                    # Extract contacts
                    c_l, c_r = utils.extract_feet_contacts(x, [3, 4], [7, 8], velfactor=0.02)
                    X.append(anim.pos[i: i+window])
                    Q.append(anim.quats[i: i+window])
                    seq_names.append(seq_name)
                    subjects.append(subjects)
                    contacts_l.append(c_l)
                    contacts_r.append(c_r)

                    i += offset

    X = np.asarray(X)
    Q = np.asarray(Q)
    contacts_l = np.asarray(contacts_l)
    contacts_r = np.asarray(contacts_r)

    # Sequences around XZ = 0
    xzs = np.mean(X[:, :, 0, ::2], axis=1, keepdims=True)
    X[:, :, 0, 0] = X[:, :, 0, 0] - xzs[..., 0]
    X[:, :, 0, 2] = X[:, :, 0, 2] - xzs[..., 1]

    # Unify facing on last seed frame
    X, Q = utils.rotate_at_frame(X, Q, anim.parents, n_past=npast)

    return X, Q, anim.parents, contacts_l, contacts_r


def get_train_stats(bvh_folder, train_set):
    """
    Extract the same training set as in the paper in order to compute the normalizing statistics
    :return: Tuple of (local position mean vector, local position standard deviation vector, local joint offsets tensor)
    """
    print('Building the train set...')
    xtrain, qtrain, parents, _, _ = get_lafan1_set(bvh_folder, train_set, window=50, offset=20)

    print('Computing stats...\n')
    # Joint offsets : are constant, so just take the first frame:
    offsets = xtrain[0:1, 0:1, 1:, :]  # Shape : (1, 1, J, 3)

    # Global representation:
    q_glbl, x_glbl = utils.quat_fk(qtrain, xtrain, parents)

    # Global positions stats:
    x_mean = np.mean(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
    x_std = np.std(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)

    return x_mean, x_std, offsets
