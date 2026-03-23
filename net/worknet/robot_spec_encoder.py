# robot_spec_encoder.py
import numpy as np
import torch
import torch.nn as nn


# -------------------------
# IO
# -------------------------
def load_robot_spec_npz(npz_path: str, device="cpu") -> dict:
    z = np.load(npz_path, allow_pickle=True)
    spec = {k: z[k] for k in z.files}

    # 转 torch（字符串保持 numpy object，不参与计算）
    to_torch = [
        "parent_idx", "axis_local", "joint_type", "q_min", "q_max", "limited", "rest_q",
        "offset_base", "offset_len", "joint_pos_base", "parent_pos_base", "edge_index",
        "base_has_freejoint",
    ]
    out = {}
    for k, v in spec.items():
        if k in to_torch:
            out[k] = torch.as_tensor(v, device=device)
        else:
            out[k] = v
    return out


# -------------------------
# Feature builders
# -------------------------
def build_robot_node_features(
    spec: dict,
    use_offset_vec: bool = True,
    use_limits: bool = True,
) -> torch.Tensor:
    """
    返回 node feature: (Nr, F)

    默认包含：
      axis_local(3)
      + offset_base(3) [可选]
      + offset_len(1)
      + q_min/q_max/rest_q/limited (2+1+1) [可选]

    joint_type 建议用 embedding 单独处理（更稳定、更表达）
    """
    axis = spec["axis_local"].float()                            # (Nr,3)
    off = spec["offset_base"].float()                            # (Nr,3)
    off_len = spec["offset_len"].float().unsqueeze(-1)           # (Nr,1)

    feats = [axis, off_len]
    if use_offset_vec:
        feats.insert(1, off)  # axis, offset, offset_len

    if use_limits:
        qmin = spec["q_min"].float().unsqueeze(-1)
        qmax = spec["q_max"].float().unsqueeze(-1)
        rest = spec["rest_q"].float().unsqueeze(-1)
        limited = spec["limited"].float().unsqueeze(-1)          # 0/1
        feats += [qmin, qmax, rest, limited]

    x = torch.cat(feats, dim=-1)  # (Nr,F)
    return x


# -------------------------
# Tree positional encoding utils
# -------------------------
@torch.no_grad()
def _build_children_list(parent_idx: torch.Tensor):
    """
    parent_idx: (Nr,) int, parent=-1 for roots
    return children: list[list[int]] in the original index order (XML order)
    """
    parent_idx = parent_idx.long()
    Nr = int(parent_idx.numel())
    children = [[] for _ in range(Nr)]
    for i in range(Nr):
        p = int(parent_idx[i].item())
        if p >= 0:
            children[p].append(i)
    return children


@torch.no_grad()
def compute_tree_positions(parent_idx: torch.Tensor,
                           max_depth: int = 63,
                           max_roots: int = 15,
                           max_sib: int = 15):
    """
    从 parent_idx 计算三种“树位置编码”：
      - depth[i]      : i 到其 root joint 的深度（root depth=0）
      - root_id[i]    : i 所属 root joint 的编号（0..R-1）
      - sibling_idx[i]: i 在其父节点 children 列表中的序号（0..deg-1）

    返回：depth, root_id, sibling_idx (均为 (Nr,) long)，并做 clamp 防御。
    """
    parent_idx = parent_idx.long()
    Nr = int(parent_idx.numel())

    # roots (can be multiple due to excluded floating base)
    roots = [i for i in range(Nr) if int(parent_idx[i].item()) < 0]
    root_map = {r: k for k, r in enumerate(roots)}

    children = _build_children_list(parent_idx)

    depth = torch.zeros(Nr, dtype=torch.long, device=parent_idx.device)
    root_id = torch.zeros(Nr, dtype=torch.long, device=parent_idx.device)
    sibling_idx = torch.zeros(Nr, dtype=torch.long, device=parent_idx.device)

    # sibling index
    for p in range(Nr):
        for s, c in enumerate(children[p]):
            sibling_idx[c] = s

    # depth & root_id by walking parents
    for i in range(Nr):
        d = 0
        cur = i
        p = int(parent_idx[cur].item())
        while p >= 0:
            d += 1
            cur = p
            p = int(parent_idx[cur].item())
            if d > max_depth:
                break
        depth[i] = min(d, max_depth)

        # cur is now the root joint index for i (or capped)
        rid = root_map.get(cur, 0)
        root_id[i] = rid

    # clamp for embedding vocab safety
    depth = depth.clamp(min=0, max=max_depth)
    root_id = root_id.clamp(min=0, max=max_roots)
    sibling_idx = sibling_idx.clamp(min=0, max=max_sib)
    return depth, root_id, sibling_idx


# -------------------------
# Encoder
# -------------------------
class RobotSpecEncoder(nn.Module):
    """
    Tree-aware RobotSpec encoder:
      - node embed: MLP(node_feat) + Emb(joint_type)
      - tree positional embeddings: depth + root_id + sibling_idx
      - bidirectional message passing on parent-child edges
      - GRU update for stability

    Output:
      - R: (Nr, D) robot joint tokens
    """

    def __init__(
        self,
        node_feat_dim: int,
        d_model: int = 128,
        num_layers: int = 4,
        joint_type_vocab: int = 16,
        joint_type_emb_dim: int = 16,
        msg_dim: int = 128,
        dropout: float = 0.1,
        # tree positional embedding vocab sizes (safe upper bounds)
        depth_max: int = 63,       # depth in [0..63]
        root_max: int = 15,        # root_id in [0..15] (multi-root safe)
        sibling_max: int = 15,     # sibling idx in [0..15]
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.depth_max = depth_max
        self.root_max = root_max
        self.sibling_max = sibling_max

        # categorical embeddings
        self.jtype_emb = nn.Embedding(joint_type_vocab, joint_type_emb_dim)

        # tree positional embeddings (this is the "tree positional embedding + depth embedding")
        self.depth_emb = nn.Embedding(depth_max + 1, d_model)
        self.root_emb = nn.Embedding(root_max + 1, d_model)
        self.sibling_emb = nn.Embedding(sibling_max + 1, d_model)

        # initial projection
        self.in_mlp = nn.Sequential(
            nn.Linear(node_feat_dim + joint_type_emb_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # message MLPs (separate directions)
        self.msg_up = nn.Sequential(
            nn.Linear(d_model, msg_dim),
            nn.GELU(),
            nn.Linear(msg_dim, d_model),
        )   # child -> parent

        self.msg_down = nn.Sequential(
            nn.Linear(d_model, msg_dim),
            nn.GELU(),
            nn.Linear(msg_dim, d_model),
        )  # parent -> child

        self.gru = nn.GRUCell(input_size=d_model * 2, hidden_size=d_model)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def _build_children(self, parent_idx: torch.Tensor):
        return _build_children_list(parent_idx)

    def forward(self, node_feat: torch.Tensor, parent_idx: torch.Tensor, joint_type: torch.Tensor):
        """
        node_feat: (Nr, F) float
        parent_idx: (Nr,) long, -1 roots
        joint_type: (Nr,) long

        return:
          R: (Nr, D)
        """
        Nr = node_feat.shape[0]
        joint_type = joint_type.long()

        # 1) base node embedding
        jt = self.jtype_emb(joint_type.clamp(min=0, max=self.jtype_emb.num_embeddings - 1))
        h = self.in_mlp(torch.cat([node_feat, jt], dim=-1))  # (Nr, D)

        # 2) tree positional embedding (depth + root_id + sibling_idx)
        depth, root_id, sib = compute_tree_positions(
            parent_idx=parent_idx,
            max_depth=self.depth_max,
            max_roots=self.root_max,
            max_sib=self.sibling_max,
        )
        h = h + self.depth_emb(depth) + self.root_emb(root_id) + self.sibling_emb(sib)
        h = self.ln(h)

        # 3) tree message passing
        children = self._build_children(parent_idx.long())

        for _ in range(self.num_layers):
            # upward messages: aggregate from children to parent
            msg_to_parent = torch.zeros_like(h)
            for p in range(Nr):
                if len(children[p]) == 0:
                    continue
                ch = torch.stack([h[c] for c in children[p]], dim=0)  # (deg, D)
                m = self.msg_up(ch).mean(dim=0)                       # (D,)
                msg_to_parent[p] = m

            # downward messages: from parent to child
            msg_to_child = torch.zeros_like(h)
            for c in range(Nr):
                p = int(parent_idx[c].item())
                if p < 0:
                    continue
                msg_to_child[c] = self.msg_down(h[p])

            # update
            inp = torch.cat([msg_to_parent + msg_to_child, h], dim=-1)  # (Nr, 2D)
            h_new = self.gru(inp, h)                                     # (Nr, D)
            h = self.ln(h + self.dropout(h_new))

        return h


# -------------------------
# Quick test
# -------------------------
if __name__ == "__main__":
    spec = load_robot_spec_npz("robot_spec_t1.npz", device="cpu")
    node_feat = build_robot_node_features(spec, use_offset_vec=True, use_limits=True)
    enc = RobotSpecEncoder(node_feat_dim=node_feat.shape[-1], d_model=256, num_layers=4)

    R = enc(node_feat, spec["parent_idx"], spec["joint_type"])
    print("Robot tokens R:", R.shape)  # (Nr, 128)
    # print first few joint names for sanity
    jn = spec["joint_names"]
    print("First 5 joints:", [jn[i] for i in range(min(5, len(jn)))])
