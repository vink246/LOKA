"""Train clipped PPO + domain randomization on the LOKA Walker MuJoCo plant.

    python -m loka.dr_rl.train_ppo
    python -m loka.dr_rl.train_ppo --seed 7
    python -m loka.dr_rl.train_ppo --no-dr --timesteps 200000
    python -m loka.dr_rl.train_ppo --vec subproc --n-envs 8
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from loka.dr_rl.config import (
    LOKA_ROOT,
    best_checkpoint_dir,
    checkpoint_dir,
    checkpoint_zip_name,
    load_dr_config,
    vecnormalize_name,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train domain-randomized PPO on the LOKA Walker (native MuJoCo).",
    )
    parser.add_argument("--config", default=None, help="Path to dr_rl/config.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--vec", choices=("dummy", "subproc"), default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for PPO, env resets, and eval (default: 0).",
    )
    parser.add_argument("--no-dr", action="store_true", help="Disable domain randomization.")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable VecNormalize on observations/rewards.",
    )
    parser.add_argument("--tb", default=None, help="TensorBoard log directory.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override checkpoint directory (default: results/dr_rl).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="SB3 device (auto, cpu, cuda).",
    )
    return parser.parse_args(argv)


def _make_env(config: dict, *, domain_randomize: bool, seed: int, rank: int):
    def _init():
        from stable_baselines3.common.monitor import Monitor

        from loka.dr_rl.env import LokaWalkerDREnv

        env = LokaWalkerDREnv(
            config=config,
            domain_randomize=domain_randomize,
            observation_noise=domain_randomize,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def _build_vec_env(
    config: dict,
    *,
    n_envs: int,
    vec_kind: str,
    domain_randomize: bool,
    seed: int,
    normalize_obs: bool,
    normalize_reward: bool,
    training: bool,
):
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    env_fns = [
        _make_env(config, domain_randomize=domain_randomize, seed=seed, rank=i)
        for i in range(n_envs)
    ]
    vec_cls = SubprocVecEnv if vec_kind == "subproc" else DummyVecEnv
    venv = vec_cls(env_fns)
    if normalize_obs or normalize_reward:
        venv = VecNormalize(
            venv,
            training=training,
            norm_obs=normalize_obs,
            norm_reward=normalize_reward and training,
            clip_obs=10.0,
            gamma=float((config.get("ppo") or {}).get("gamma", 0.99)),
        )
    return venv


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_dr_config(args.config)
    ppo_cfg = dict(config.get("ppo") or {})

    n_envs = int(args.n_envs or ppo_cfg.get("n_envs", 4))
    vec_kind = args.vec or str(ppo_cfg.get("vec_env", "dummy"))
    timesteps = int(args.timesteps or ppo_cfg.get("n_timesteps", 1_000_000))
    domain_randomize = not args.no_dr
    normalize_obs = bool(ppo_cfg.get("normalize_obs", True)) and not args.no_normalize
    normalize_reward = bool(ppo_cfg.get("normalize_reward", True)) and not args.no_normalize

    out_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir(config)
    if not out_dir.is_absolute():
        out_dir = LOKA_ROOT / out_dir
    best_dir = best_checkpoint_dir(config)
    if args.output_dir:
        best_dir = out_dir / "best"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    src_cfg = Path(str(config.get("_config_path") or ""))
    if src_cfg.is_file():
        shutil.copy2(src_cfg, out_dir / "config.yaml")

    zip_name = checkpoint_zip_name(config)
    vec_name = vecnormalize_name(config)

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from torch import nn

    from loka.dr_rl.policy import save_vec_stats

    train_env = _build_vec_env(
        config,
        n_envs=n_envs,
        vec_kind=vec_kind,
        domain_randomize=domain_randomize,
        seed=args.seed,
        normalize_obs=normalize_obs,
        normalize_reward=normalize_reward,
        training=True,
    )
    eval_env = _build_vec_env(
        config,
        n_envs=1,
        vec_kind="dummy",
        domain_randomize=False,
        seed=args.seed + 10_000,
        normalize_obs=normalize_obs,
        normalize_reward=False,
        training=False,
    )

    net_arch = list(ppo_cfg.get("net_arch") or [256, 256])
    policy_kwargs = {
        "net_arch": dict(pi=net_arch, vf=net_arch),
        "activation_fn": nn.ReLU,
        "log_std_init": float(ppo_cfg.get("log_std_init", -0.5)),
    }

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=int(ppo_cfg.get("n_steps", 512)),
        batch_size=int(ppo_cfg.get("batch_size", 64)),
        n_epochs=int(ppo_cfg.get("n_epochs", 20)),
        learning_rate=float(ppo_cfg.get("learning_rate", 5.05e-5)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.1)),
        ent_coef=float(ppo_cfg.get("ent_coef", 5e-3)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.87)),
        max_grad_norm=float(ppo_cfg.get("max_grad_norm", 1.0)),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=args.tb,
    )

    class SaveVecNormalizeCallback(BaseCallback):
        def __init__(self, save_path: Path):
            super().__init__()
            self.save_path = Path(save_path)

        def _on_step(self) -> bool:
            save_vec_stats(self.model.get_env(), self.save_path)
            return True

    class DomainRandomizationCurriculum(BaseCallback):
        """Keep the plant nominal until ``warmup_steps``, then turn DR on."""

        def __init__(self, warmup_steps: int, enabled: bool):
            super().__init__()
            self.warmup_steps = int(warmup_steps)
            self.enabled = bool(enabled) and self.warmup_steps > 0
            self._pending = self.enabled

        def _set_dr(self, flag: bool) -> None:
            self.training_env.env_method("set_domain_randomize", bool(flag))

        def _on_training_start(self) -> None:
            if self.enabled:
                self._set_dr(False)

        def _on_step(self) -> bool:
            if self._pending and self.num_timesteps >= self.warmup_steps:
                self._set_dr(True)
                self._pending = False
                print(
                    f"[dr_rl] enabling domain randomization at "
                    f"step {self.num_timesteps}"
                )
            return True

    eval_freq = max(int(ppo_cfg.get("eval_freq", 16384)) // n_envs, 1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(out_dir / "eval"),
        eval_freq=eval_freq,
        n_eval_episodes=int(ppo_cfg.get("n_eval_episodes", 5)),
        deterministic=True,
        callback_on_new_best=SaveVecNormalizeCallback(best_dir / vec_name),
    )
    name_prefix = str(
        (config.get("checkpoint") or {}).get("name_prefix", "ppo_walker")
    )
    ckpt_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=str(out_dir / "checkpoints"),
        name_prefix=name_prefix,
        save_replay_buffer=False,
        save_vecnormalize=normalize_obs or normalize_reward,
    )
    warmup_steps = int(ppo_cfg.get("nominal_warmup_steps", 0)) if domain_randomize else 0
    dr_cb = DomainRandomizationCurriculum(warmup_steps, enabled=domain_randomize)

    tag = str((config.get("checkpoint") or {}).get("name_prefix", "ppo_walker"))
    print(
        f"[dr_rl] training {tag} on {train_env.num_envs} env(s) "
        f"seed={args.seed} dr={domain_randomize} warmup={warmup_steps} "
        f"vec={vec_kind} steps={timesteps} out={out_dir} "
        f"config={config.get('_config_path')}"
    )
    model.learn(
        total_timesteps=timesteps,
        callback=CallbackList([ckpt_cb, eval_cb, dr_cb]),
        progress_bar=False,
    )

    final_zip = out_dir / zip_name
    model.save(str(final_zip))
    save_vec_stats(train_env, out_dir / vec_name)
    best_zip = best_dir / "best_model.zip"
    if best_zip.is_file():
        shutil.copy2(best_zip, best_dir / zip_name)
    train_env.close()
    eval_env.close()
    print(f"[dr_rl] saved {final_zip}")
    print(
        "[dr_rl] evaluate with: python -m loka.run_walker_suite "
        "--baselines dr_rl --tests ice,backpack,shove_back,dead_right_hip"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
