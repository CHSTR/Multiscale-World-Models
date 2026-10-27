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

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
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
        x: (B*T, D)
        """
        return self.net(x)


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
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
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
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x

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
