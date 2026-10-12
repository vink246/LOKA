"""The unit of work shared by the local pool and Slurm: one trial of one condition."""

from __future__ import annotations

from dataclasses import dataclass

from loka.harness.stochastic import trial_seed


@dataclass(frozen=True)
class WorkUnit:
    task: str
    perturbation: str
    baseline: str
    trial: int
    seed: int

    @property
    def id(self) -> str:
        return (
            f"{self.task}__{self.perturbation}__{self.baseline}"
            f"__trial_{self.trial:02d}"
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task": self.task,
            "perturbation": self.perturbation,
            "baseline": self.baseline,
            "trial": int(self.trial),
            "seed": int(self.seed),
        }


def trial_indices(suite) -> list[int]:
    if suite.num_trials < 1:
        raise ValueError("num_trials must be >= 1")
    if suite.trial is None:
        return list(range(suite.num_trials))
    if suite.trial < 0 or suite.trial >= suite.num_trials:
        raise ValueError(f"trial must be in 0..{suite.num_trials - 1}")
    return [suite.trial]


def iter_units(suite) -> list[WorkUnit]:
    """Stable order: task, perturbation, baseline, trial."""
    units: list[WorkUnit] = []
    for task in suite.tasks:
        for perturbation in task.perturbations:
            for baseline in suite.baselines:
                for trial in trial_indices(suite):
                    units.append(
                        WorkUnit(
                            task=task.name,
                            perturbation=perturbation.name,
                            baseline=baseline.name,
                            trial=int(trial),
                            seed=trial_seed(suite.seed, trial),
                        )
                    )
    return units
