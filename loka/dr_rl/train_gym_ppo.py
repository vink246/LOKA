"""Gym / Isaac-style PPO harness (no gait clock). Does not touch results/dr_rl/.

    python -m loka.dr_rl.train_gym_ppo
    python -m loka.dr_rl.train_gym_ppo --seed 0
    python -m loka.dr_rl.play --config loka/dr_rl/config_gym.yaml \\
        --checkpoint results/dr_rl_gym/best
"""

from __future__ import annotations

import sys

from loka.dr_rl.config import GYM_CONFIG_PATH
from loka.dr_rl.train_ppo import main as train_main


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--config" not in argv:
        argv = ["--config", str(GYM_CONFIG_PATH), *argv]
    return train_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
