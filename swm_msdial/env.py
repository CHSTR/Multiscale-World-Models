"""
MultiScaleDial: entorno sintético para aislar el efecto de la información multiescala
en world models latentes (LeWM / JEPA).

Adaptado de la implementación de referencia
(/home/chr/Documents/multiescala_env/msdial_env.py) según el plan
docs/superpowers/plans/2026-11-18-msdial-integration.md (Task 1):
goal + `_set_state`/`_set_goal_state`, `terminated` al alcanzar la meta
(tolerancias eps_p/eps_th del spec §3) y flags `ever_pos`/`ever_dial`
acumulados por episodio (los lee la eval para el éxito por componente).

Estado:   p ∈ [R, 1-R]^2 (posición del cuerpo, escala gruesa)
          θ_k ∈ S^1 para cada nivel k en `levels` (diales anidados, escala ~ 2^-k)
Acción:   a ∈ [-1, 1]^(2 + K)  -> (dx, dy, dθ_1, ..., dθ_K)
Obs:      {"pixels": uint8 HxWx3, "state": float32 [p, sin θ, cos θ]}
Info:     {"state": float64 [p, θ] crudo (para callables/controlador),
          "success_pos"/"success_dial": bool del paso,
          "ever_pos"/"ever_dial": bool OR acumulado del episodio,
          "dpos": float, "dth_max": float} (None si no hay goal fijado)

El dial de nivel k es un anillo [r_k/2, r_k] con r_k = 0.9 * R * 2^-(k-1) y una cuña
de 60° que marca θ_k. Con img_size=224 y R=0.12: anillos de 24/12/6/3 px de radio
exterior para k=1..4 (k=3 cabe en un patch de 16 px; k=4 es sub-patch).

La única variable que cambia entre variantes de nivel es la escala visual del dial:
dinámica, dimensión de acción y cuerpo son idénticos.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs import registration
from scipy.ndimage import gaussian_filter

# Intensidades (gris; se replican a RGB para no dar atajos de color)
BG, BODY, RING, WEDGE = 0.55, 0.30, 0.85, 0.10
RING_B, DELTA_B = 0.60, 0.35  # estilo "balanced"
WEDGE_HALF = np.deg2rad(30.0)

# Tolerancias de éxito (spec §3)
EPS_P = 0.05
EPS_TH = 0.3  # rad


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def register():
    """Registra swm/MultiScaleDial-v0 (idempotente)."""
    if "swm/MultiScaleDial-v0" not in registration.registry:
        registration.register(
            id="swm/MultiScaleDial-v0",
            entry_point="swm_msdial.env:MultiScaleDialEnv",
        )


class MultiScaleDialEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        levels=(2,),
        img_size=224,
        supersample=4,
        body_radius=0.12,
        pos_step=0.03,
        rot_step=0.25,
        sensor_noise=2.0 / 255,
        distractor_band=None,      # sigma (px) del pasabanda DoG del fondo, o None
        distractor_amp=0.08,
        distractor_dynamic=False,  # si True, el fondo deriva aleatoriamente (impredecible)
        max_steps=200,
        dial_style="balanced",     # "wedge" (cuña oscura) | "balanced" (momento dipolar nulo)
        render_mode="rgb_array",
    ):
        super().__init__()
        self.dial_style = dial_style
        self.levels = tuple(sorted(levels))
        self.K = len(self.levels)
        self.H = img_size
        self.ss = supersample
        self.R = body_radius
        self.pos_step, self.rot_step = pos_step, rot_step
        self.sensor_noise = sensor_noise
        self.distractor_band = distractor_band
        self.distractor_amp = distractor_amp
        self.distractor_dynamic = distractor_dynamic
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.Box(-1.0, 1.0, (2 + self.K,), np.float32)
        self.observation_space = spaces.Dict(
            pixels=spaces.Box(0, 255, (self.H, self.H, 3), np.uint8),
            state=spaces.Box(-np.inf, np.inf, (2 + 2 * self.K,), np.float32),
        )

        # Grilla supersampleada local (bounding box del cuerpo), precomputada
        half = int(np.ceil(self.R * self.H)) + 2
        self._half = half
        n = 2 * half * self.ss
        c = (np.arange(n) + 0.5) / self.ss - half  # coords en px relativos al centro del bbox
        self._gx, self._gy = np.meshgrid(c, c)

        self.pos = np.full(2, 0.5)
        self.theta = np.zeros(self.K)
        self._bg = None
        self._bg_offset = np.zeros(2, int)
        self.t = 0
        # goal fijado por la eval vía `_set_goal_state` (spec §3/§6.2)
        self.goal = None
        self._ever_pos = False
        self._ever_dial = False

    # ---------- estado ----------
    def get_state(self):
        return np.concatenate([self.pos, self.theta]).astype(np.float64)

    def set_state(self, state):
        state = np.asarray(state, np.float64)
        self.pos = np.clip(state[:2], self.R, 1 - self.R)
        self.theta = wrap(state[2:2 + self.K])

    def _set_state(self, state):
        """Alias estilo TwoRoom para los callables de la eval (spec §6.2)."""
        self.set_state(state)

    def _set_goal_state(self, goal_state):
        """Fija la meta del episodio (pos clipeada, θ envuelto)."""
        g = np.asarray(goal_state, np.float64)
        goal = np.empty(2 + self.K)
        goal[:2] = np.clip(g[:2], self.R, 1 - self.R)
        goal[2:] = wrap(g[2:2 + self.K])
        self.goal = goal

    def sample_state(self, rng=None):
        rng = rng or self.np_random
        pos = rng.uniform(self.R, 1 - self.R, 2)
        th = rng.uniform(-np.pi, np.pi, self.K)
        return np.concatenate([pos, th])

    def _obs(self):
        st = np.concatenate([self.pos, np.sin(self.theta), np.cos(self.theta)]).astype(np.float32)
        return {"pixels": self.render(), "state": st}

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.set_state(options.get("state", self.sample_state()))
        self.t = 0
        self.goal = None
        self._ever_pos = False
        self._ever_dial = False
        self._make_background()
        return self._obs(), {"state": self.get_state()}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), -1, 1)
        self.pos = np.clip(self.pos + self.pos_step * a[:2], self.R, 1 - self.R)
        self.theta = wrap(self.theta + self.rot_step * a[2:])
        if self.distractor_dynamic:
            self._bg_offset += self.np_random.integers(-3, 4, 2)
        self.t += 1
        trunc = self.t >= self.max_steps
        if self.goal is not None:
            dpos = float(np.linalg.norm(self.pos - self.goal[:2]))
            dth = np.abs(wrap(self.theta - self.goal[2:]))
            dth_max = float(dth.max()) if self.K else 0.0
            ok_p = dpos < EPS_P
            ok_th = bool(dth_max < EPS_TH)
            self._ever_pos = self._ever_pos or ok_p
            self._ever_dial = self._ever_dial or ok_th
            terminated = bool(ok_p and ok_th)
        else:
            dpos = dth_max = None
            ok_p = ok_th = False
            terminated = False
        info = {
            "state": self.get_state(),
            "success_pos": bool(ok_p),
            "success_dial": bool(ok_th),
            "ever_pos": bool(self._ever_pos),
            "ever_dial": bool(self._ever_dial),
            "dpos": dpos,
            "dth_max": dth_max,
        }
        return self._obs(), 0.0, terminated, trunc, info

    # ---------- render ----------
    def _make_background(self):
        bg = np.full((self.H, self.H), BG, np.float64)
        if self.distractor_band is not None:
            s = self.distractor_band
            w = self.np_random.standard_normal((self.H, self.H))
            band = gaussian_filter(w, s, mode="wrap") - gaussian_filter(w, 2 * s, mode="wrap")
            band *= self.distractor_amp / (band.std() + 1e-8)
            self._band = band
        else:
            self._band = None
        self._bg = bg

    def _body_patch(self):
        """Renderiza el cuerpo + diales en el bbox local, supersampleado y con box-filter."""
        px = self.pos * self.H
        # offset subpíxel del centro respecto del bbox
        cx0, cy0 = np.floor(px).astype(int)
        fx, fy = px - np.floor(px)
        dx = self._gx - (fx - 0.5)
        dy = self._gy - (fy - 0.5)
        r = np.hypot(dx, dy)
        phi = np.arctan2(dy, dx)
        Rp = self.R * self.H

        val = np.full(r.shape, np.nan)
        val[r <= Rp] = BODY
        for k, th in zip(self.levels, self.theta):
            ro = 0.9 * Rp * 2.0 ** (-(k - 1))
            ring = (r <= ro) & (r >= ro / 2)
            wedge = ring & (np.abs(wrap(phi - th)) <= WEDGE_HALF)
            if self.dial_style == "wedge":
                val[ring] = RING
                val[wedge] = WEDGE
            else:
                # Cuña partida radialmente: mitad interior clara (+δ), exterior oscura (-δ).
                # Con r_m = ro*(9/16)^(1/3), ∫ r^2 f(r) dr = 0  -> el dipolo (primer término
                # dependiente de θ que sobrevive a un pasa-bajos) se anula; la fuga a escalas
                # gruesas pasa a ser O((ro/σ)^2) menor.
                rm = ro * (9 / 16) ** (1 / 3)
                val[ring] = RING_B
                val[wedge & (r < rm)] = RING_B + DELTA_B
                val[wedge & (r >= rm)] = RING_B - DELTA_B

        alpha = (~np.isnan(val)).astype(np.float64)
        val = np.nan_to_num(val)
        n = val.shape[0] // self.ss
        down = lambda x: x.reshape(n, self.ss, n, self.ss).mean((1, 3))
        return down(val * alpha), down(alpha), (cx0, cy0)

    def render(self):
        img = self._bg.copy()
        if self._band is not None:
            img += np.roll(self._band, tuple(self._bg_offset), (0, 1))
        pv, pa, (cx, cy) = self._body_patch()
        h = self._half
        y0, x0 = cy - h + 1, cx - h + 1
        # recorte por si el bbox toca el borde
        ys, xs = slice(max(y0, 0), min(y0 + 2 * h, self.H)), slice(max(x0, 0), min(x0 + 2 * h, self.H))
        py, pxs = slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0)
        a = pa[py, pxs]
        img[ys, xs] = img[ys, xs] * (1 - a) + pv[py, pxs]
        if self.sensor_noise > 0:
            img = img + self.np_random.normal(0, self.sensor_noise, img.shape)
        img = np.clip(np.round(img * 255), 0, 255).astype(np.uint8)
        return np.repeat(img[..., None], 3, axis=2)

    # ---------- evaluación ----------
    def success(self, state, goal, eps_p=0.05, eps_th=0.3):
        """Devuelve dict con éxito por componente (para el diseño 2x2 coarse/fine)."""
        state, goal = np.asarray(state), np.asarray(goal)
        ok_p = np.linalg.norm(state[:2] - goal[:2]) < eps_p
        dth = np.abs(wrap(state[2:] - goal[2:]))
        ok_th = bool(np.all(dth < eps_th))
        return {"pos": bool(ok_p), "dial": ok_th, "all": bool(ok_p and ok_th),
                "dpos": float(np.linalg.norm(state[:2] - goal[:2])), "dth": dth.tolist()}

    def sample_goal(self, start, mode, rng=None):
        """mode ∈ {'coarse', 'fine', 'both'}: qué componente difiere del inicio."""
        rng = rng or self.np_random
        g = np.array(start, np.float64)
        if mode in ("coarse", "both"):
            while True:
                cand = rng.uniform(self.R, 1 - self.R, 2)
                if np.linalg.norm(cand - g[:2]) > 0.3:
                    g[:2] = cand
                    break
        if mode in ("fine", "both"):
            d = rng.uniform(np.pi / 2, np.pi, self.K) * rng.choice([-1, 1], self.K)
            g[2:] = wrap(g[2:] + d)
        return g


def scripted_policy(state, target, rng, noise=0.3, gain_p=8.0, gain_th=1.5):
    """Controlador proporcional ruidoso hacia un waypoint (p, θ). Para recolectar datos."""
    dp = gain_p * (target[:2] - state[:2])
    dth = gain_th * wrap(target[2:] - state[2:])
    a = np.concatenate([dp, dth])
    a = np.clip(a, -1, 1) + noise * rng.standard_normal(a.shape)
    return np.clip(a, -1, 1).astype(np.float32)
