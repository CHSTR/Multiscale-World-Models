"""Tests en CPU del front-end multiescala y los diagnósticos (sin pesos ni datos reales).

    pip install pytest && python -m pytest tests -q

Los backbones DINO se construyen con pesos aleatorios guardados en un archivo temporal (se pasan por
``ckpt_path``), así que no hay descargas. En DINOv3 se ponen los LayerScale en 1 para que la red
sea sensible a la entrada (con la init aleatoria de 1e-5 la salida casi no depende de la imagen).
"""

import importlib.util
import math
import subprocess
import sys
import types
import uuid
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from wavelet.frontend import (  # noqa: E402
    BandTransform, ablate_bands, band_names, n_bands, resolve_band_indices, sums_to_input,
)

IMG = 112


def smooth_images(n=2, size=IMG, seed=0):
    """Imágenes 'naturales' de juguete: ruido pasa-bajos (la energía vive en las escalas gruesas)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, size // 8, size // 8, generator=g)
    return F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)


# --------------------------------------------------------------------------------------
# Transformadas y ablación
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["starlet", "rgb_repeat", "random"])
@pytest.mark.parametrize("levels", [1, 3])
def test_bands_sum_to_input(kind, levels):
    x = torch.randn(2, 3, 64, 64, dtype=torch.float64)
    bt = BandTransform(kind, levels).double()
    bands = bt(x)
    assert bands.shape == (2, n_bands(kind, levels), 3, 64, 64)
    assert sums_to_input(kind)
    assert torch.allclose(bands.sum(1), x, atol=1e-10)
    assert len(bt.names) == bands.shape[1]


@pytest.mark.parametrize("merge", [False, True])
def test_swt_shapes(merge):
    x = torch.randn(2, 3, 64, 64)
    bands = BandTransform("swt", 3, merge_orientations=merge)(x)
    assert bands.shape[1] == (4 if merge else 10)
    assert not sums_to_input("swt")
    assert band_names("swt", 3, merge)[-1] == "LL3"


def test_random_filters_deterministic_and_zero_mean():
    a = BandTransform("random", 3, random_seed=7).random_kernels
    b = BandTransform("random", 3, random_seed=7).random_kernels
    assert torch.equal(a, b)
    assert torch.allclose(a.flatten(1).sum(1), torch.zeros(3), atol=1e-6)


def test_resolve_and_ablate():
    names = band_names("starlet", 3)
    assert names == ["w1", "w2", "w3", "c3"]
    assert resolve_band_indices(["w1", "c3"], names) == [0, 3]
    assert resolve_band_indices(["details"], names) == [0, 1, 2]
    assert resolve_band_indices([-1], names) == [3]
    with pytest.raises(ValueError):
        resolve_band_indices(["w9"], names)

    bands = torch.randn(4, 4, 3, 16, 16)
    z = ablate_bands(bands, [1], "zero")
    assert torch.equal(z[:, 1], torch.zeros_like(z[:, 1])) and torch.equal(z[:, 0], bands[:, 0])
    m = ablate_bands(bands, [3], "mean")
    assert torch.allclose(m[:, 3].mean((-2, -1)), bands[:, 3].mean((-2, -1)), atol=1e-6)
    assert torch.allclose(m[:, 3].std((-2, -1)), torch.zeros(4, 3), atol=1e-6)
    n = ablate_bands(bands, [2], "noise", torch.Generator().manual_seed(0))
    assert not torch.allclose(n[:, 2], bands[:, 2])
    assert torch.allclose(n[:, 2].std((-2, -1)), bands[:, 2].std((-2, -1)), rtol=0.35)


# --------------------------------------------------------------------------------------
# Encoders DINO
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def dino_ckpts(tmp_path_factory):
    d = tmp_path_factory.mktemp("ck")
    torch.hub.set_dir(str(d / "hub"))  # evita reusar archivos cacheados con el mismo nombre
    from src.dinov2.hub.backbones import dinov2_vits14
    from src.dinov3.hub.backbones import dinov3_vits16

    torch.manual_seed(0)
    p2 = d / f"v2_{uuid.uuid4().hex}.pth"
    torch.save(dinov2_vits14(pretrained=False, img_size=224).state_dict(), p2)
    m3 = dinov3_vits16(pretrained=False)
    for name, p in m3.named_parameters():
        if name.endswith("gamma"):
            p.data.fill_(1.0)
    p3 = d / f"v3_{uuid.uuid4().hex}.pth"
    torch.save(m3.state_dict(), p3)
    return {"v2": str(p2), "v3": str(p3)}


def make_encoder(ver, ckpts, **kw):
    from src.encoders.dinov2_encoder import DinoV2Encoder
    from src.encoders.dinov3_encoder import DinoV3Encoder

    torch.manual_seed(1)
    cls = DinoV2Encoder if ver == "v2" else DinoV3Encoder
    return cls(ckpt_path=ckpts[ver], lora_dropout=0.0, **kw).eval()


def cls_out(enc, x):
    with torch.no_grad():
        return enc(x).last_hidden_state[:, 0]


def patch_tokens(enc, x):
    from src.encoders.vit_lora import _find_patch_embed

    with torch.no_grad():
        return _find_patch_embed(enc.model)(enc._frontend_input(x))


@pytest.mark.parametrize("ver", ["v2", "v3"])
def test_no_cross_sample_leakage_in_batch(dino_ckpts, ver):
    """Regresión: el CLS de una imagen no puede depender de las otras imágenes del batch.
    (En DINOv2 un ``[0]`` en Block.attn_residual_func copiaba la atención de la imagen 0 a todas.)"""
    enc = make_encoder(ver, dino_ckpts)
    a, b = smooth_images(1, seed=1), smooth_images(1, seed=2)
    alone = cls_out(enc, b)[0]
    in_batch = cls_out(enc, torch.cat([a, b]))
    assert torch.allclose(alone, in_batch[1], atol=1e-4)
    assert not torch.allclose(in_batch[0], in_batch[1], atol=1e-3)


EXACT_AT_INIT = [
    dict(starlet_levels=3, patch_init="equivalent"),
    dict(starlet_levels=3, fusion="residual"),
    dict(frontend="swt", frontend_levels=3, fusion="residual"),
    dict(frontend="swt", frontend_levels=2, swt_merge_orientations=True, fusion="residual"),
    dict(frontend="rgb_repeat", frontend_levels=3, patch_init="equivalent"),
    dict(frontend="random", frontend_levels=3, patch_init="equivalent"),
]


@pytest.mark.parametrize("ver", ["v2", "v3"])
@pytest.mark.parametrize("kw", EXACT_AT_INIT, ids=lambda k: "-".join(f"{a}={b}" for a, b in k.items()))
def test_new_inits_reproduce_pretrained(dino_ckpts, ver, kw):
    x = smooth_images()
    ref = cls_out(make_encoder(ver, dino_ckpts), x)
    out = cls_out(make_encoder(ver, dino_ckpts, **kw), x)
    assert (out - ref).abs().max() < 1e-4


@pytest.mark.parametrize("ver", ["v2", "v3"])
def test_legacy_init_does_not_reproduce_pretrained(dino_ckpts, ver):
    """Documenta el problema: con la init original el patch embedding ve otra imagen en el paso 0."""
    x = smooth_images()
    pe_ref = patch_tokens(make_encoder(ver, dino_ckpts), x)
    pe_leg = patch_tokens(make_encoder(ver, dino_ckpts, starlet_levels=3), x)
    pe_eq = patch_tokens(make_encoder(ver, dino_ckpts, starlet_levels=3, patch_init="equivalent"), x)
    rel_leg = ((pe_leg - pe_ref).norm() / pe_ref.norm()).item()
    rel_eq = ((pe_eq - pe_ref).norm() / pe_ref.norm()).item()
    assert rel_eq < 1e-5
    assert rel_leg > 0.1, rel_leg


def _load_original_module(relpath, name):
    """Carga la versión de dino-lora (antes de este cambio) de un archivo del repo."""
    try:
        src = subprocess.run(["git", "-C", str(ROOT), "show", f"origin/dino-lora:{relpath}"],
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        pytest.skip("no hay rama origin/dino-lora para comparar")
    mod = types.ModuleType(name)
    mod.__file__ = str(ROOT / relpath)
    exec(compile(src, relpath, "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("ver", ["v2", "v3"])
@pytest.mark.parametrize("levels", [0, 3])
def test_legacy_defaults_match_original_code(dino_ckpts, ver, levels):
    """Con los argumentos por defecto, mismas claves de state_dict y misma salida que el código
    original: los checkpoints existentes cargan y se comportan igual."""
    rel = f"src/encoders/dinov{ver[1]}_encoder.py"
    old_mod = _load_original_module(rel, f"old_dinov{ver[1]}")
    OldCls = getattr(old_mod, f"DinoV{ver[1]}Encoder")
    torch.manual_seed(1)
    old = OldCls(ckpt_path=dino_ckpts[ver], starlet_levels=levels, lora_dropout=0.0).eval()
    new = make_encoder(ver, dino_ckpts, starlet_levels=levels)
    sd_old, sd_new = old.state_dict(), new.state_dict()
    assert sd_old.keys() == sd_new.keys()
    assert all(sd_old[k].shape == sd_new[k].shape for k in sd_old)
    new.load_state_dict(sd_old)  # = cargar un checkpoint viejo
    x = smooth_images()
    assert torch.allclose(cls_out(old, x), cls_out(new, x), atol=1e-5)


@pytest.mark.parametrize("ver", ["v2", "v3"])
def test_band_ablation_hooks(dino_ckpts, ver):
    enc = make_encoder(ver, dino_ckpts, starlet_levels=3, patch_init="equivalent")
    x = smooth_images()
    clean = cls_out(enc, x)
    assert enc.band_names == ["w1", "w2", "w3", "c3"]
    idx = enc.set_band_ablation(["c3"], mode="zero")
    assert idx == [3]
    ablated = cls_out(enc, x)
    assert not torch.allclose(ablated, clean, atol=1e-4)
    enc.clear_band_ablation()
    assert torch.allclose(cls_out(enc, x), clean, atol=1e-6)
    # band_scale = 1 no cambia nada; = 0 equivale a quitar todas las bandas
    enc.set_band_scale(torch.ones(4))
    assert torch.allclose(cls_out(enc, x), clean, atol=1e-5)
    enc.set_band_scale(None)


def test_residual_branch_trains_and_reports_ratio(dino_ckpts):
    enc = make_encoder("v3", dino_ckpts, starlet_levels=3, fusion="residual").train()
    x = smooth_images()
    out = enc(x).last_hidden_state[:, 0].pow(2).mean()
    out.backward()
    from src.encoders.vit_lora import _find_patch_embed

    proj = _find_patch_embed(enc.model).proj
    assert proj.ms.weight.grad is not None and proj.ms.weight.grad.abs().sum() > 0
    stats = enc.frontend_stats()
    assert "ms_branch_ratio" in stats and float(stats["ms_branch_ratio"]) == 0.0
    assert set(k for k in stats if k.startswith("band_weight/")) == {
        "band_weight/w1", "band_weight/w2", "band_weight/w3", "band_weight/c3"}


def test_configs_instantiate(dino_ckpts):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    cases = [("dinov3_swt_lora", "v3", 33), ("dinov2_swt_lora", "v2", 33),
             ("dinov3_starlet_lora", "v3", 12), ("dinov3_lora", "v3", 3)]
    for model, ver, chans in cases:
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config/train")):
            cfg = compose(config_name="lewm", overrides=[f"model={model}",
                                                         f"model.encoder.ckpt_path={dino_ckpts[ver]}"])
        enc = instantiate(cfg.model.encoder)
        assert enc.in_chans == chans, (model, enc.in_chans)


# --------------------------------------------------------------------------------------
# Diagnóstico de bandas con un JEPA pequeño
# --------------------------------------------------------------------------------------
def test_band_importance_batch_and_ig_completeness(dino_ckpts):
    from diagnostics.common import band_importance_batch
    from jepa import JEPA
    from module import MLP, ARPredictor, Embedder

    torch.manual_seed(0)
    enc = make_encoder("v2", dino_ckpts, starlet_levels=2, patch_init="equivalent")
    model = JEPA(
        encoder=enc,
        predictor=ARPredictor(num_frames=2, depth=1, heads=2, mlp_dim=64, input_dim=32,
                              hidden_dim=32, output_dim=32, dim_head=16),
        action_encoder=Embedder(input_dim=4, emb_dim=32),
        projector=MLP(input_dim=384, hidden_dim=64, output_dim=32),
        pred_proj=MLP(input_dim=32, hidden_dim=64, output_dim=32),
    ).eval()
    model.requires_grad_(False)
    B, T = 2, 3
    batch = {"pixels": smooth_images(B * T, size=IMG).view(B, T, 3, IMG, IMG),
             "action": torch.randn(B, T, 4)}
    out = band_importance_batch(model, batch, ctx_len=2, n_preds=1, modes=("zero", "noise"), ig_steps=64)
    assert out["names"] == ["w1", "w2", "c2"]
    assert ("c2", "zero") in out["ablations"] and ("details", "noise") in out["ablations"]
    gap = float(out["loss_all"] - out["loss_none"])
    assert math.isclose(float(out["ig"].sum()), gap, rel_tol=0.03, abs_tol=1e-6), (out["ig"].sum(), gap)
    assert enc._band_ablation is None and enc._band_scale is None  # deja el encoder limpio


def test_rollout_errors_shapes_and_teacher_forcing(dino_ckpts):
    from diagnostics.rollout_error import rollout_errors
    from jepa import JEPA
    from module import MLP, ARPredictor, Embedder

    torch.manual_seed(0)
    enc = make_encoder("v2", dino_ckpts)
    model = JEPA(
        encoder=enc,
        predictor=ARPredictor(num_frames=2, depth=1, heads=2, mlp_dim=64, input_dim=32,
                              hidden_dim=32, output_dim=32, dim_head=16),
        action_encoder=Embedder(input_dim=4, emb_dim=32),
        projector=MLP(input_dim=384, hidden_dim=64, output_dim=32),
        pred_proj=MLP(input_dim=32, hidden_dim=64, output_dim=32),
    ).eval()
    B, T = 2, 5
    batch = {"pixels": smooth_images(B * T, size=IMG).view(B, T, 3, IMG, IMG),
             "action": torch.randn(B, T, 4)}
    mse, cos, var = rollout_errors(model, batch, history=2, horizon=3)
    assert mse.shape == (2, 3) and cos.shape == (2, 3) and var > 0
    # El primer paso del rollout es exactamente la predicción de entrenamiento (teacher forcing).
    from diagnostics.common import encode_pixels, pred_loss_per_sample

    with torch.no_grad():
        z = encode_pixels(model, batch["pixels"][:, :3])
        pred = model.predict(z[:, :2], model.action_encoder(batch["action"][:, :2]))[:, -1:]
        first = (pred - z[:, 2:3]).pow(2).mean(dim=(1, 2))
    assert torch.allclose(first, mse[:, 0], atol=1e-6)


# --------------------------------------------------------------------------------------
# Estadística, probes, perturbaciones, generador de comandos
# --------------------------------------------------------------------------------------
def test_stats_core():
    from scipy.stats import binomtest

    from diagnostics.stats_core import compare, holm, mcnemar_exact, paired_sign_flip, wilson_ci

    for b, c in [(3, 9), (0, 5), (10, 10), (1, 0)]:
        assert math.isclose(mcnemar_exact(b, c), binomtest(min(b, c), b + c, 0.5).pvalue, rel_tol=1e-9)
    lo, hi = wilson_ci(111, 150)
    assert lo < 0.74 < hi and 0.65 < lo and hi < 0.81
    d = np.array([1, 1, 1, 1, 1, 1, 0, 0, -1])
    assert paired_sign_flip(d) == pytest.approx(mcnemar_exact(6, 1), rel=1e-9)
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

    keys = [(42, e, 0) for e in range(150)]
    rng = np.random.default_rng(0)
    a = rng.random(150) < 0.74
    b = a.copy()
    b[:17] = False  # 17 episodios que la variante pierde
    ref = {k: [float(v)] for k, v in zip(keys, a)}
    oth = {k: [float(v)] for k, v in zip(keys, b)}
    r = compare(ref, oth)
    assert r["delta_pp"] < 0 and "p_mcnemar" in r and r["only_other"] == 0


def test_probe_core():
    from diagnostics.probe_core import episode_split, fit_probe

    rng = np.random.default_rng(0)
    n, d = 3000, 16
    ep = np.repeat(np.arange(60), n // 60)
    X = rng.normal(size=(n, d))
    theta = rng.uniform(-np.pi, np.pi, size=n)
    Xf = np.concatenate([X, np.sin(theta)[:, None], np.cos(theta)[:, None]], 1)
    Y = np.stack([X @ rng.normal(size=d), theta], 1)
    res = fit_probe(Xf, Y, ep, angle_dims=[1], names=["lin", "ang"])
    assert res["names"] == ["lin", "ang_sin", "ang_cos"]
    assert np.all(res["r2_test"] > 0.99)
    assert res["angle_mae_deg"]["ang"] < 2.0
    tr, te = episode_split(ep, 0.2, 0)
    assert not set(ep[tr]) & set(ep[te])


def test_perturbations():
    from diagnostics.perturb import ImagePerturbation

    x = torch.rand(3, 64, 64)
    hf = lambda im: (im - F.avg_pool2d(im[None], 3, 1, 1)[0]).pow(2).mean()  # noqa: E731
    assert hf(ImagePerturbation("blur", 2.0)(x)) < 0.2 * hf(x)
    assert hf(ImagePerturbation("downsample", 4)(x)) < hf(x)
    n1 = ImagePerturbation("noise", 0.1, seed=3)(x)
    n2 = ImagePerturbation("noise", 0.1, seed=3)(x)
    assert torch.equal(n1, n2) and 0 <= n1.min() and n1.max() <= 1
    assert torch.equal(ImagePerturbation("none")(x), x)


def test_matrix_generates_commands():
    out = subprocess.run([sys.executable, str(ROOT / "experiments/matrix.py"), "--envs", "cube",
                          "--encoders", "dinov3", "--groups", "core", "controls",
                          "--stages", "train", "eval", "ablate", "stats", "--seeds", "0"],
                         capture_output=True, text=True, check=True).stdout
    assert "model.encoder.patch_init=equivalent" in out
    assert "'ablation.bands=[LH1,HL1,HH1]'" in out
    assert "diagnostics.paired_stats" in out and "--ref dinov3-rgb" in out


def test_eval_configs_have_diagnostic_keys():
    from omegaconf import OmegaConf

    for name in ("pusht", "tworoom", "cube"):
        cfg = OmegaConf.load(ROOT / f"config/eval/{name}.yaml")
        assert cfg.ablation.bands is None and cfg.perturb.kind == "none" and "tag" in cfg.output
