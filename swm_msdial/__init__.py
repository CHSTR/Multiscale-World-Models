"""MultiScaleDial — registro gym + exports.

`import swm_msdial` registra el entorno `swm/MultiScaleDial-v0` en gymnasium.
No modifica el paquete stable_worldmodel instalado: el registro vive en este
repo y lo consumen nuestros scripts (train/eval) al importar el paquete.
"""
from .env import MultiScaleDialEnv, scripted_policy, register

register()

__all__ = ["MultiScaleDialEnv", "scripted_policy", "register"]
