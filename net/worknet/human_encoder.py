# human_encoder.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """
    Optional conditioning by morphology m_h.
    Produce (gamma, beta) to modulate hidden states: h <- (1+gamma)*h + beta
    """
    def __init__(self, cond_dim: int, d_model: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2 * d_model),
        )

    def forward(self, m_h: torch.Tensor):
        # m_h: (B, cond_dim)
        gb = self.net(m_h)  # (B, 2D)
        gamma, beta = gb.chunk(2, dim=-1)  # (B, D), (B, D)
        return gamma, beta


class FeedForward(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FactorizedSTBlock(nn.Module):
    """
    One block = Spatial MHSA (per time step) + Temporal MHSA (per joint) + FFN
    Tokens layout: (B, L, N, D)
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1, mlp_ratio: float = 4.0,
                 use_film: bool = False):
        super().__init__()
        self.use_film = use_film

        self.ln_spa = nn.LayerNorm(d_model)
        self.spa_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ln_tmp = nn.LayerNorm(d_model)
        self.tmp_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ln_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, mlp_ratio=mlp_ratio, dropout=dropout)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, film_gamma=None, film_beta=None):
        """
        x: (B,L,N,D)
        film_gamma/beta: (B,D) if provided
        """
        B, L, N, D = x.shape

        # ---- Spatial attn: for each t, attend across N tokens ----
        xs = self.ln_spa(x)
        if self.use_film and (film_gamma is not None) and (film_beta is not None):
            # broadcast (B,1,1,D)
            xs = (1.0 + film_gamma[:, None, None, :]) * xs + film_beta[:, None, None, :]

        xs = xs.reshape(B * L, N, D)  # (B*L, N, D)
        out, _ = self.spa_attn(xs, xs, xs, need_weights=False)
        out = out.reshape(B, L, N, D)
        x = x + self.drop(out)

        # ---- Temporal attn: for each token, attend across L frames ----
        xt = self.ln_tmp(x)
        if self.use_film and (film_gamma is not None) and (film_beta is not None):
            xt = (1.0 + film_gamma[:, None, None, :]) * xt + film_beta[:, None, None, :]

        xt = xt.permute(0, 2, 1, 3).reshape(B * N, L, D)  # (B*N, L, D)
        out, _ = self.tmp_attn(xt, xt, xt, need_weights=False)
        out = out.reshape(B, N, L, D).permute(0, 2, 1, 3)  # back to (B,L,N,D)
        x = x + self.drop(out)

        # ---- FFN ----
        xf = self.ln_ffn(x)
        if self.use_film and (film_gamma is not None) and (film_beta is not None):
            xf = (1.0 + film_gamma[:, None, None, :]) * xf + film_beta[:, None, None, :]

        x = x + self.ffn(xf)
        return x


class HumanEncoder(nn.Module):
    """
    Inputs:
      x_joint: (B,L,Nh,Dj)
      x_root : (B,L,Dr)
      m_h    : (B,13) optional

    We build tokens per time step:
      tokens[t] = [root_token(t), joint_1(t), ..., joint_Nh(t)]  => N = Nh+1

    Outputs:
      H_all  : (B,L,Nh+1,D)
      H_root : (B,L,D)
      H_joint: (B,L,Nh,D)
    """
    def __init__(
        self,
        Nh: int = 14,
        Dj: int = 12,
        Dr: int = 12,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        max_len: int = 1024,
        use_morph_film: bool = True,
        morph_dim: int = 13,
    ):
        super().__init__()
        self.Nh = Nh
        self.Dj = Dj
        self.Dr = Dr
        self.d_model = d_model
        self.use_morph_film = use_morph_film

        # input projection
        self.joint_proj = nn.Linear(Dj, d_model)
        self.root_proj = nn.Linear(Dr, d_model)

        # token (joint-id) embedding: include root token id=0, joints id=1..Nh
        self.token_emb = nn.Embedding(Nh + 1, d_model)

        # temporal positional embedding (learnable)
        self.time_emb = nn.Embedding(max_len, d_model)

        self.in_ln = nn.LayerNorm(d_model)
        self.in_drop = nn.Dropout(dropout)

        # morphology conditioning
        self.film = FiLM(morph_dim, d_model) if use_morph_film else None

        # ST blocks
        self.blocks = nn.ModuleList([
            FactorizedSTBlock(d_model, n_heads=n_heads, dropout=dropout, mlp_ratio=mlp_ratio,
                              use_film=use_morph_film)
            for _ in range(n_layers)
        ])

        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, x_joint: torch.Tensor, x_root: torch.Tensor, m_h: torch.Tensor = None):
        """
        x_joint: (B,L,Nh,Dj)
        x_root : (B,L,Dr)
        m_h    : (B,13) optional
        """
        if x_joint.dim() == 3:
            x_joint = x_joint.unsqueeze(0)
        if x_root.dim() == 2:
            x_root = x_root.unsqueeze(0)

        B, L, Nh, Dj = x_joint.shape
        assert Nh == self.Nh and Dj == self.Dj, f"Expected x_joint (B,L,{self.Nh},{self.Dj}), got {x_joint.shape}"
        assert x_root.shape[0] == B and x_root.shape[1] == L and x_root.shape[2] == self.Dr, \
            f"Expected x_root (B,L,{self.Dr}), got {x_root.shape}"

        # project
        hj = self.joint_proj(x_joint)                 # (B,L,Nh,D)
        hr = self.root_proj(x_root).unsqueeze(2)      # (B,L,1,D)

        # concat root token first
        x = torch.cat([hr, hj], dim=2)                # (B,L,Nh+1,D)

        # add token embeddings
        token_ids = torch.arange(self.Nh + 1, device=x.device).long()  # (Nh+1,)
        tok = self.token_emb(token_ids)[None, None, :, :]              # (1,1,N,D)
        x = x + tok

        # add temporal embeddings
        t_ids = torch.arange(L, device=x.device).long().clamp(max=self.time_emb.num_embeddings - 1)
        te = self.time_emb(t_ids)[None, :, None, :]                    # (1,L,1,D)
        x = x + te

        x = self.in_drop(self.in_ln(x))

        # morphology FiLM params
        gamma = beta = None
        if self.use_morph_film:
            if m_h is None:
                # allow running without morphology
                gamma = beta = None
            else:
                if m_h.dim() == 1:
                    m_h = m_h.unsqueeze(0)
                gamma, beta = self.film(m_h)  # (B,D),(B,D)

        # ST blocks
        for blk in self.blocks:
            x = blk(x, film_gamma=gamma, film_beta=beta)

        x = self.out_ln(x)

        H_root = x[:, :, 0, :]          # (B,L,D)
        H_joint = x[:, :, 1:, :]        # (B,L,Nh,D)
        return x, H_root, H_joint


# -------------------------
# Quick check
# -------------------------
if __name__ == "__main__":
    B, L, Nh = 2, 64, 14
    Dj, Dr = 12, 12
    x_joint = torch.randn(B, L, Nh, Dj)
    x_root = torch.randn(B, L, Dr)
    m_h = torch.randn(B, 13)

    enc = HumanEncoder(Nh=Nh, Dj=Dj, Dr=Dr, d_model=256, n_heads=4, n_layers=4, use_morph_film=True)
    H_all, H_root, H_joint = enc(x_joint, x_root, m_h)
    print(H_all.shape, H_root.shape, H_joint.shape)
