"""Tests del entorno MultiScaleDial (Task 1 del plan msdial).

Correr: ./.venv_hf/bin/python -m pytest tests/test_msdial_env.py -q
"""
import numpy as np
import pytest
import gymnasium as gym


@pytest.fixture()
def env2():
    import swm_msdial  # registra swm/MultiScaleDial-v0
    from swm_msdial.env import MultiScaleDialEnv

    e = MultiScaleDialEnv(levels=(2,))
    e.reset(seed=0)
    return e


def test_register_and_make():
    import swm_msdial  # noqa: F401 — el import registra el id

    env = gym.make("swm/MultiScaleDial-v0", levels=(2,), max_episode_steps=100)
    obs, info = env.reset(seed=0)
    assert obs["state"].shape == (4,)          # [p, sinθ, cosθ] con K=1 (levels=(2,))
    assert env.render().shape == (224, 224, 3)  # pixels via render (wrapper)
    assert "agent_state" in info
    env.close()


def test_set_state_roundtrip(env2):
    # θ fuera de [-π, π] debe envolverse; pos fuera de [R, 1-R] debe clipearse
    s = np.array([0.01, 0.99, 7.0, -7.5])
    env2._set_state(s)
    st = env2.get_state()
    assert st[0] == pytest.approx(0.12, abs=1e-6)   # clip a R
    assert st[1] == pytest.approx(0.88, abs=1e-6)   # clip a 1-R
    for got, want in zip(st[2:], [(7.0 + np.pi) % (2 * np.pi) - np.pi,
                                  (-7.5 + np.pi) % (2 * np.pi) - np.pi]):
        assert got == pytest.approx(want, abs=1e-6)


def test_step_terminates_only_with_goal(env2):
    from swm_msdial.env import scripted_policy

    rng = np.random.default_rng(0)
    # sin goal: nunca termina
    for _ in range(30):
        a = scripted_policy(env2.get_state(), env2.sample_state(rng), rng)
        _, _, term, _, _ = env2.step(a)
        assert term is False
    # con goal: termina en <= 50 pasos con el controlador oráculo
    goal = env2.sample_state(rng)
    env2._set_goal_state(goal)
    terminated = False
    for i in range(50):
        a = scripted_policy(env2.get_state(), goal, rng, noise=0.0)
        _, _, term, _, info = env2.step(a)
        if i == 0:
            # convención TwoRoom: distancia real en el info para las trazas
            assert isinstance(info["distance_to_target"], float)
            assert info["distance_to_target"] > 0
        terminated = terminated or term
    assert terminated is True


def test_goal_boundary_wrap(env2):
    # meta θ=π-0.05 y estado θ=-π+0.05 están a 0.1 rad, no a π-0.1
    env2._set_state(np.array([0.5, 0.5, -np.pi + 0.05, 0.0]))
    env2._set_goal_state(np.array([0.5, 0.5, np.pi - 0.05, 0.0]))
    _, _, term, _, info = env2.step(np.zeros(4))
    assert info["success_dial"] is True
    assert info["dth_max"] < 0.3


def test_render_shape_dtype(env2):
    img = env2.render()
    assert img.shape == (224, 224, 3)
    assert img.dtype == np.uint8


def test_sample_goal_modes(env2):
    rng = np.random.default_rng(1)
    start = env2.sample_state(rng)
    g = env2.sample_goal(start, "coarse", rng)
    assert np.linalg.norm(g[:2] - start[:2]) > 0.3
    assert np.allclose(g[2:], start[2:])
    g = env2.sample_goal(start, "fine", rng)
    assert np.allclose(g[:2], start[:2])
    dth = np.abs((g[2:] - start[2:] + np.pi) % (2 * np.pi) - np.pi)
    assert np.all((dth >= np.pi / 2) & (dth <= np.pi))
    g = env2.sample_goal(start, "both", rng)
    assert np.linalg.norm(g[:2] - start[:2]) > 0.3
    assert np.any(g[2:] != start[2:])


def test_ever_flags_reset(env2):
    # alcanza el goal -> ever_* True; reset() limpia
    from swm_msdial.env import scripted_policy

    rng = np.random.default_rng(2)
    goal = env2.sample_state(rng)
    env2._set_goal_state(goal)
    for _ in range(50):
        _, _, term, _, info = env2.step(
            scripted_policy(env2.get_state(), goal, rng, noise=0.0))
        if term:
            break
    assert info["ever_pos"] is True and info["ever_dial"] is True
    env2.reset(seed=1)
    assert env2._ever_pos is False and env2._ever_dial is False


def test_oracle_reachability():
    from swm_msdial.env import MultiScaleDialEnv, scripted_policy

    rng = np.random.default_rng(0)
    for mode in ("coarse", "fine", "both"):
        env = MultiScaleDialEnv(levels=(2,))
        res = []
        for i in range(20):
            _, info = env.reset(seed=i)
            goal = env.sample_goal(info["agent_state"], mode, rng)
            env._set_goal_state(goal)
            st = info["agent_state"]
            for _ in range(50):
                _, _, _, _, info = env.step(
                    scripted_policy(st, goal, rng, noise=0.0))
                st = info["agent_state"]
            res.append(env.success(st, goal)["all"])
        assert np.mean(res) == 1.0, f"oracle falló en modo {mode}"
