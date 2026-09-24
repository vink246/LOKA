"""Load a frozen Stable-Baselines3 PPO policy for suite evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from loka.dr_rl.config import load_dr_config, resolve_checkpoint_paths


def save_vec_stats(vec_env, path: Path) -> None:
    """Write running obs stats without pickling the VecEnv."""
    import pickle

    if not hasattr(vec_env, "obs_rms"):
        return
    payload = {
        "mean": np.asarray(vec_env.obs_rms.mean),
        "var": np.asarray(vec_env.obs_rms.var),
        "count": float(getattr(vec_env.obs_rms, "count", 1.0)),
        "clip_obs": float(getattr(vec_env, "clip_obs", 10.0)),
        "epsilon": float(getattr(vec_env, "epsilon", 1e-8)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def _load_vec_stats(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load mean/var from a slim dict or an SB3 VecNormalize pickle."""
    import pickle

    with path.open("rb") as handle:
        saved = pickle.load(handle)
    clip_obs = 10.0
    epsilon = 1e-8
    if isinstance(saved, dict) and "mean" in saved:
        mean = np.asarray(saved["mean"], dtype=np.float32)
        var = np.asarray(saved["var"], dtype=np.float32)
        clip_obs = float(saved.get("clip_obs", clip_obs))
        epsilon = float(saved.get("epsilon", epsilon))
        return mean, var, clip_obs, epsilon
    rms = getattr(saved, "obs_rms", saved)
    mean = np.asarray(rms.mean, dtype=np.float32)
    var = np.asarray(rms.var, dtype=np.float32)
    clip_obs = float(getattr(saved, "clip_obs", clip_obs))
    epsilon = float(getattr(saved, "epsilon", epsilon))
    return mean, var, clip_obs, epsilon


class FrozenPPOPolicy:
    """Deterministic PPO actor with optional VecNormalize obs stats."""

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        *,
        config: dict | None = None,
    ):
        from stable_baselines3 import PPO

        cfg = config if config is not None else load_dr_config()
        policy_path, vec_path = resolve_checkpoint_paths(cfg, checkpoint=checkpoint)
        if not policy_path.is_file():
            raise FileNotFoundError(
                f"No DR-RL checkpoint at {policy_path}. "
                "Train one with: python -m loka.dr_rl.train_ppo"
            )
        self.policy_path = policy_path
        self.vecnormalize_path = vec_path
        self.model = PPO.load(str(policy_path), device="cpu")
        self._obs_mean = None
        self._obs_var = None
        self._clip_obs = 10.0
        self._epsilon = 1e-8
        if vec_path is not None:
            mean, var, clip_obs, epsilon = _load_vec_stats(vec_path)
            self._obs_mean = mean
            self._obs_var = var
            self._clip_obs = clip_obs
            self._epsilon = epsilon

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        if self._obs_mean is None or self._obs_var is None:
            return obs
        normed = (obs - self._obs_mean) / np.sqrt(self._obs_var + self._epsilon)
        return np.clip(normed, -self._clip_obs, self._clip_obs).astype(np.float32)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        obs = self._normalize(obs)
        action, _ = self.model.predict(obs, deterministic=True)
        return np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
