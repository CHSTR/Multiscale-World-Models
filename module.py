import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj, weights=None):
        """
        proj: (T, B, D)
        weights: (B,) opcional. Pesos por muestra (p. ej. IS del replay
            priorizado): la función característica empírica pasa a ser una media
            ponderada autonormalizada, así que el estadístico mide la isotropía
            bajo la distribución de los datos aunque el batch venga sesgado.
            ``None`` = comportamiento original.
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        if weights is None:
            err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        else:
            w = weights.to(device=x_t.device, dtype=torch.float32)
            w = (w / w.sum()).to(x_t.dtype).view(1, -1, 1, 1)
            err = ((x_t.cos() * w).sum(-3) - self.phi).square() + (x_t.sin() * w).sum(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time


class SVReg(nn.Module):
    """Sliced Variance Regularization (unilateral std >= 1 floor over S^{D-1})."""

    def __init__(self, num_proj: int = 1024, eps: float = 1e-8, pooled: bool = False):
        super().__init__()
        self.num_proj = int(num_proj)
        self.eps = float(eps)
        self.pooled = bool(pooled)

    def forward(self, proj: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """
        proj: (T, N, D) (or (N, D) treated as T=1)
        weights: optional (N,) importance-sampling weights along the sample axis.
        """
        if proj.ndim == 2:
            proj = proj.unsqueeze(0)
        if proj.ndim != 3:
            raise ValueError(f"SVReg expects 2D or 3D tensor, got shape {tuple(proj.shape)}")
        T, N, D = proj.shape
        if proj.numel() == 0 or T == 0 or N == 0:
            return proj.new_zeros((), dtype=torch.float32)

        with torch.autocast(device_type=proj.device.type, enabled=False):
            x = proj.float()
            w_norm = None
            if weights is not None:
                if weights.ndim != 1 or weights.numel() != N:
                    raise ValueError(
                        f"SVReg weights must have shape ({N},), got {tuple(weights.shape)}"
                    )
                w = weights.to(device=x.device, dtype=torch.float32).clamp_min(0.0)
                w_sum = w.sum()
                w_norm = torch.where(
                    w_sum > 0,
                    w / w_sum.clamp_min(1e-12),
                    torch.full_like(w, 1.0 / max(1, N)),
                )

            if self.pooled:
                x = x.reshape(1, T * N, D)
                if w_norm is not None:
                    w_norm = (w_norm.unsqueeze(0).expand(T, N) / float(T)).reshape(-1)

            A = torch.randn(D, self.num_proj, device=x.device, dtype=torch.float32)
            A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
            p = x @ A  # (T', N', num_proj)

            if w_norm is None:
                mean = p.mean(dim=-2, keepdim=True)
                var = (p - mean).square().mean(dim=-2)
            else:
                w_view = w_norm.view(1, -1, 1)
                mean = (p * w_view).sum(dim=-2, keepdim=True)
                var = ((p - mean).square() * w_view).sum(dim=-2)

            std = torch.sqrt(var.clamp_min(0.0) + self.eps)
            return F.relu(1.0 - std).square().mean()


class SlownessLoss(nn.Module):
    """Variance-normalized temporal slowness on the last 1/8 of latent dimensions."""

    def __init__(self, slow_frac: float = 0.125, eps: float = 1e-6):
        super().__init__()
        self.slow_frac = float(slow_frac)
        self.eps = float(eps)

    def forward(self, emb: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        """
        emb: (B, T, D) or (B, T, N, D)
        weights: optional (B,) importance-sampling weights along the batch axis.
        """
        if emb.ndim not in (3, 4):
            raise ValueError(f"SlownessLoss expects 3D or 4D tensor, got shape {tuple(emb.shape)}")
        if emb.numel() == 0 or emb.size(0) == 0 or emb.size(1) < 2:
            return emb.new_zeros((), dtype=torch.float32)
        if emb.ndim == 4 and emb.size(2) == 0:
            return emb.new_zeros((), dtype=torch.float32)

        B, T = emb.shape[0], emb.shape[1]
        D = emb.shape[-1]
        d_slow = max(1, int(D * self.slow_frac))

        with torch.autocast(device_type=emb.device.type, enabled=False):
            b = emb[..., -d_slow:].float()
            if weights is not None:
                if weights.ndim != 1 or weights.numel() != B:
                    raise ValueError(
                        f"SlownessLoss weights must have shape ({B},), got {tuple(weights.shape)}"
                    )
                w = weights.to(device=b.device, dtype=torch.float32).clamp_min(0.0)
                w_sum = w.sum()
                w_norm = torch.where(
                    w_sum > 0,
                    w / w_sum.clamp_min(1e-12),
                    torch.full_like(w, 1.0 / max(1, B)),
                )
                if b.ndim == 3:
                    diff_sq = (b[:, 1:] - b[:, :-1]).square().sum(dim=-1).mean(dim=1)
                    num = (w_norm * diff_sq).sum()
                    w_bt = w_norm.view(B, 1, 1) / float(T)
                    mu = (w_bt * b).sum(dim=(0, 1), keepdim=True)
                    tr_var = (w_bt * (b - mu).square()).sum()
                else:
                    N = b.shape[2]
                    diff_sq = (b[:, 1:] - b[:, :-1]).square().sum(dim=-1).mean(dim=(1, 2))
                    num = (w_norm * diff_sq).sum()
                    w_btn = w_norm.view(B, 1, 1, 1) / float(T * N)
                    mu = (w_btn * b).sum(dim=(0, 1, 2), keepdim=True)
                    tr_var = (w_btn * (b - mu).square()).sum()
            else:
                num = (b[:, 1:] - b[:, :-1]).square().sum(dim=-1).mean()
                tr_var = b.reshape(-1, d_slow).var(dim=0, unbiased=False).sum()

            denom = (2.0 * tr_var).clamp_min(self.eps)
            return num / denom
    
def sigreg_input(emb, max_tokens=None):
    """Prepara el input de SIGReg, ``(T, n_samples, D)``.

    * ``emb`` ``(B, T, D)`` (JEPA global): igual que antes, ``emb.transpose(0, 1)``.
    * ``emb`` ``(B, T, N, D)`` (JEPA espacial): cada patch token es una muestra.
      Se submuestrean ``max_tokens`` tokens del pool ``B*N`` (mismos índices para
      todo t) porque el costo de SIGReg escala con ``n_samples * num_proj``.
      Ojo: el estadístico Epps-Pulley escala con ``n_samples``; con
      ``max_tokens = batch_size`` el peso ``lambda`` tiene la misma escala que en
      el baseline.
    """
    if emb.ndim == 3:
        return emb.transpose(0, 1)
    B, T, N, D = emb.shape
    x = emb.permute(1, 0, 2, 3).reshape(T, B * N, D)
    if max_tokens and B * N > max_tokens:
        idx = torch.randperm(B * N, device=emb.device)[:max_tokens]
        x = x[:, idx]
    return x


class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class GroupNorm1d(nn.GroupNorm):
    """GroupNorm adapter accepting (num_channels, num_groups=8, ...) for MLP & Hydra _partial_."""

    def __init__(
        self,
        num_channels: int,
        num_groups: int = 8,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim <= 2:
            return super().forward(x)
        lead = x.shape[:-1]
        return super().forward(x.reshape(-1, x.shape[-1])).reshape(*lead, -1)


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (..., D)
        """
        if x.ndim <= 2:
            return self.net(x)
        lead = x.shape[:-1]
        return self.net(x.reshape(-1, x.shape[-1])).reshape(*lead, -1)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        gated_residual: bool = False,
        gate_init: float = 0.1,
    ):
        super().__init__()
        out_dim = output_dim or input_dim
        self.gated_residual = bool(gated_residual)
        if self.gated_residual:
            if out_dim != input_dim:
                raise ValueError(
                    f"ARPredictor gated_residual requires output_dim ({out_dim}) == input_dim ({input_dim})"
                )
            self.res_gate = nn.Parameter(torch.full((out_dim,), float(gate_init)))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            out_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        x_in = x
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        out = self.transformer(x, c)
        if self.gated_residual:
            return x_in + self.res_gate * out
        return out

# Version adaptada para JEPA
# Se utilizó como referencia y se reutilizaron cosas de: https://github.com/zhechencai/WaveViT/blob/main/classification/wavevit.py#L159
class WaveletAttention(nn.Module):
    """Atención Wave-ViT: comprime K/V via DWT Haar a N/4 tokens y aporta
    detalles finos por un residual IDWT de alta frecuencia.

    Resuelve el dilema coste/detalle: la matriz de atención se reduce de
    (N, N) a (N, N/4) usando sub-bandas wavelet, mientras que la IDWT del
    V original reinyecta los bordes/texturas que la compresión atenúa.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim debe ser divisible por num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5
        # Proyecciones QKV y salida
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # ponytail: conv 4C->C estándar; depthwise+pointwise (Conv2d(4C,4C,3,groups=4C)+Conv2d(4C,C,1))
        # sería ~10x menos params, migrar si la huella de memoria aprieta.
        self.fuse = nn.Conv2d(dim * 4, dim, kernel_size=3, stride=1, padding=1)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        # Filtros Haar fijos (no entrenables): paso-bajo [1,1]/sqrt(2), paso-alto [1,-1]/sqrt(2)
        s2 = 2.0 ** 0.5
        lo = torch.tensor([1.0, 1.0]) / s2
        hi = torch.tensor([1.0, -1.0]) / s2
        # lo_w/hi_w aplican sobre ancho (kh=1, kw=2); lo_h/hi_h sobre alto (kh=2, kw=1)
        self.register_buffer("lo_w", lo.view(1, 1, 1, 2))
        self.register_buffer("hi_w", hi.view(1, 1, 1, 2))
        self.register_buffer("lo_h", lo.view(1, 1, 2, 1))
        self.register_buffer("hi_h", hi.view(1, 1, 2, 1))

    def _dwt(self, x):
        """DWT Haar 2D separable. x: (B, C, H, W) -> 4 sub-bandas (B, C, H/2, W/2)."""
        B, C, H, W = x.shape
        lo_w = self.lo_w.expand(C, 1, 1, 2)
        hi_w = self.hi_w.expand(C, 1, 1, 2)
        lo_h = self.lo_h.expand(C, 1, 2, 1)
        hi_h = self.hi_h.expand(C, 1, 2, 1)
        # Filas (ancho) con stride 2: L = paso-bajo, H_ = paso-alto
        L = F.conv2d(x, lo_w, stride=(1, 2), groups=C)  # (B, C, H, W/2)
        H_ = F.conv2d(x, hi_w, stride=(1, 2), groups=C)
        # Columnas (alto) con stride 2 sobre L y H_
        LL = F.conv2d(L, lo_h, stride=(2, 1), groups=C)  # (B, C, H/2, W/2)
        LH = F.conv2d(L, hi_h, stride=(2, 1), groups=C)  # detalle vertical de L
        HL = F.conv2d(H_, lo_h, stride=(2, 1), groups=C)  # detalle horizontal de H
        HH = F.conv2d(H_, hi_h, stride=(2, 1), groups=C)  # diagonal
        return LL, LH, HL, HH

    def _idwt(self, LL, LH, HL, HH):
        """IDWT Haar 2D: reconstruye (B, C, 2Hh, 2Wh) a partir de las 4 sub-bandas."""
        B, C, Hh, Wh = LL.shape
        lo_w = self.lo_w.expand(C, 1, 1, 2)
        hi_w = self.hi_w.expand(C, 1, 1, 2)
        lo_h = self.lo_h.expand(C, 1, 2, 1)
        hi_h = self.hi_h.expand(C, 1, 2, 1)
        # Inverso columnas: L (low-bandas original de filas) y H (high-bandas)
        L = F.conv_transpose2d(LL, lo_h, stride=(2, 1), groups=C) \
            + F.conv_transpose2d(LH, hi_h, stride=(2, 1), groups=C)  # (B, C, 2Hh, Wh)
        H = F.conv_transpose2d(HL, lo_h, stride=(2, 1), groups=C) \
            + F.conv_transpose2d(HH, hi_h, stride=(2, 1), groups=C)
        # Inverso filas: combina L (low) y H (high) -> x
        x = F.conv_transpose2d(L, lo_w, stride=(1, 2), groups=C) \
            + F.conv_transpose2d(H, hi_w, stride=(1, 2), groups=C)  # (B, C, 2Hh, 2Wh)
        return x

    def forward(self, x, output_attentions=False):
        """x: (B, N, C) con N = H*W (cuadrado perfecto; padding reflectante si H/W impares)."""
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"N={N} debe ser H*W (cuadrado perfecto)"
        # Paso 1: QKV. Q: (B, N, C), K/V -> (B, C, H, W)
        qkv = self.qkv(x).reshape(B, N, 3, C)
        q, k, v = qkv.unbind(dim=2)  # (B, N, C)
        k = k.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        # Padding reflectante para pares
        pad_w = W % 2
        pad_h = H % 2
        if pad_h or pad_w:
            k = F.pad(k, (0, pad_w, 0, pad_h), mode="reflect")
            v = F.pad(v, (0, pad_w, 0, pad_h), mode="reflect")
        # Paso 2: DWT Haar
        LL_k, LH_k, HL_k, HH_k = self._dwt(k)
        LL_v, LH_v, HL_v, HH_v = self._dwt(v)
        # Paso 3: fusión 4C -> C, aplanado a (B, Np, C) con Np = N/4 (+padding)
        k_cat = torch.cat([LL_k, LH_k, HL_k, HH_k], dim=1)  # (B, 4C, Hp, Wp)
        v_cat = torch.cat([LL_v, LH_v, HL_v, HH_v], dim=1)
        k_comp = self.fuse(k_cat)  # (B, C, Hp, Wp)
        v_comp = self.fuse(v_cat)
        Hp, Wp = k_comp.shape[-2:]
        Np = Hp * Wp
        k_comp = k_comp.reshape(B, C, Np).transpose(1, 2).contiguous()  # (B, Np, C)
        v_comp = v_comp.reshape(B, C, Np).transpose(1, 2).contiguous()
        # Multi-cabeza: (B, h, *, dim_head)
        q = q.reshape(B, N, self.num_heads, self.dim_head).transpose(1, 2)
        k_comp = k_comp.reshape(B, Np, self.num_heads, self.dim_head).transpose(1, 2)
        v_comp = v_comp.reshape(B, Np, self.num_heads, self.dim_head).transpose(1, 2)
        # Paso 4: atención asimétrica (N x Np), 4x menos coste que la estándar
        attn = (q @ k_comp.transpose(-2, -1)) * self.scale  # (B, h, N, Np)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        o_attn = (attn @ v_comp).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        # Paso 5: IDWT sobre sub-bandas originales de V -> V_recon (residual de alta resolución)
        v_recon = self._idwt(LL_v, LH_v, HL_v, HH_v)  # (B, C, 2Hp, 2Wp)
        if pad_h or pad_w:
            v_recon = v_recon[..., :H, :W]
        v_recon = v_recon.reshape(B, C, N).transpose(1, 2).contiguous()  # (B, N, C)
        # Paso 6: salida con residual interno (V_recon + O_attn) + proj/dropout + residual externo (x)
        out = self.proj(v_recon + o_attn)
        out = self.proj_drop(out)
        out = out + x
        if output_attentions:
            # ponytail: Np puede diferir de N/4 si H/W impares; el caller usa psize_w para el reshape.
            return out, attn
        return out


if __name__ == "__main__":
    # Ponytail self-check: forma, reconstrucción IDWT(DWT(x))==x y backward
    import torch
    m = WaveletAttention(dim=64, num_heads=4)
    m.eval()
    # 1) Forma par
    x = torch.randn(2, 196, 64)  # N=196 = 14*14
    y = m(x)
    assert y.shape == x.shape, f"shape mismatch: {y.shape}"
    print("forward par OK:", y.shape)
    # 2) Reconstrucción Haar perfecta
    k = torch.randn(1, 64, 8, 8)
    LL, LH, HL, HH = m._dwt(k)
    r = m._idwt(LL, LH, HL, HH)
    err = (r - k).abs().max().item()
    assert err < 1e-5, f"IDWT/DWT err {err}"
    print("reconstruccion Haar OK, err =", err)
    # 3) Forma impar (15x15 -> padding -> recorte)
    xx = torch.randn(2, 225, 64)
    yy = m(xx)
    assert yy.shape == xx.shape, f"shape impar: {yy.shape}"
    print("forward impar OK:", yy.shape)
    # 4) Backward
    m.train()
    x = torch.randn(2, 196, 64, requires_grad=True)
    loss = m(x).sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    print("backward OK")
    print("WaveletAttention self-check completo")
