"""Effort-bias streak. A failure episode opens only when this streak reaches
``consecutive_steps`` in ``loka/config/loka.yaml``.
"""

from __future__ import annotations

from collections import deque
import sys
import numpy as np

from loka.agent.loka_config import loka_config
from loka.control.robot import QVEL_JOINT0

_COST_NAMES = ("roll", "pitch", "yaw", "x", "y", "z", "wx", "wy", "wz", "vx", "vy", "vz")


def belief_actuator_deviation(plant_bias, belief_bias, contact_jacobian) -> np.ndarray:
    """Plant bias minus belief bias, in actuator torques.

    Torso mass and COM land on the floating base, not on the leg-joint entries
    of ``qfrc_bias``. The base wrench is resolved through the sole Jacobian,
    ``J_base.T @ f = Δ_base``, and the actuator value is ``J_joint.T @ f`` plus
    the joint residual. The result is zero when the two models agree.
    """
    plant = np.asarray(plant_bias, dtype=float).reshape(-1)
    belief = np.asarray(belief_bias, dtype=float).reshape(-1)
    delta = plant - belief
    joints = delta[QVEL_JOINT0:]
    jac = np.asarray(contact_jacobian, dtype=float)
    if jac.size == 0 or joints.size == 0:
        return joints
    j_base = jac[:, :QVEL_JOINT0]
    j_joint = jac[:, QVEL_JOINT0:QVEL_JOINT0 + joints.size]
    forces, *_ = np.linalg.lstsq(j_base.T, delta[:QVEL_JOINT0], rcond=None)
    return j_joint.T @ forces + joints


class EffortDeviationObserver:
    def __init__(self, window_size_steps: int | None = None):
        bias = loka_config()["effort_bias"]
        if window_size_steps is None:
            window_size_steps = int(bias["stand_window_steps"])
        self.window_size = window_size_steps
        self.torque_deviations = deque(maxlen=window_size_steps)
        self.costs = deque(maxlen=window_size_steps)
        self.names = None
        self.printout = False
        self.is_walking = False
        self.dt = 0.002
        self.step_period = 0.70
        self.consecutive_steps = int(bias["consecutive_steps"])
        self.stand_threshold_nm = float(bias["stand_threshold_nm"])
        self.walk_threshold_nm = float(bias["walk_threshold_nm"])
        self.clear_below_s = float(bias.get("clear_below_s", 0.5))
        self.consecutive_out_of_bounds = 0
        self.consecutive_in_bounds = 0
        self.joint_bias = None
        self.avg_cost = None
        self._walk = deque()
        self._was_walking = False
        self._line_width = 0
        self._named = False
        self.xyz = None

    def update(self, delta_tau, cost, printout=False, xyz=None):
        # Signed actuator image of plant bias minus belief bias.
        delta_tau = np.asarray(delta_tau, dtype=float)
        self.torque_deviations.append(delta_tau)
        self.costs.append(np.asarray(cost, dtype=float))

        if self.is_walking:
            # Mean over two step periods. Uniform dt makes the mean the integral mean.
            n = max(1, int(round(2.0 * self.step_period / self.dt)))
            self._walk.append(delta_tau)
            while len(self._walk) > n:
                self._walk.popleft()
            joint_bias = np.mean(np.stack(self._walk), axis=0)
        else:
            self._walk.clear()
            joint_bias = np.mean(np.stack(self.torque_deviations), axis=0)
        avg_cost = np.mean(np.stack(self.costs), axis=0)
        total_bias = float(np.sum(np.abs(joint_bias)))
        if self.is_walking != self._was_walking:
            self.consecutive_out_of_bounds = 0
            self.consecutive_in_bounds = 0
            self._was_walking = self.is_walking
        threshold = self.walk_threshold_nm if self.is_walking else self.stand_threshold_nm
        if total_bias > threshold:
            self.consecutive_out_of_bounds += 1
            self.consecutive_in_bounds = 0
        else:
            self.consecutive_out_of_bounds = 0
            self.consecutive_in_bounds += 1
        self.joint_bias = joint_bias
        self.avg_cost = avg_cost
        if xyz is not None:
            self.xyz = np.asarray(xyz, dtype=float).reshape(3)

        if printout:
            self._write(avg_cost, joint_bias, total_bias)

        return total_bias, joint_bias, avg_cost

    def _write(self, avg_cost, joint_bias, total_bias) -> None:
        if not self._named:
            sys.stdout.write(
                "[effort] cost order: " + " ".join(_COST_NAMES) + "\n"
            )
            self._named = True
        costs = np.atleast_1d(np.asarray(avg_cost, dtype=float))
        vals = " ".join(f"{float(costs[i]):5.2f}" for i in range(min(12, costs.size)))
        names = self.names or tuple(str(i) for i in range(len(joint_bias)))
        worst = int(np.argmax(np.abs(joint_bias))) if len(joint_bias) else 0
        worst_txt = (
            f"{names[worst]}={float(joint_bias[worst]):+.1f}"
            if len(joint_bias)
            else ""
        )
        pos = ""
        if self.xyz is not None:
            pos = (
                f"  xyz={self.xyz[0]:+.3f} {self.xyz[1]:+.3f} {self.xyz[2]:+.3f}"
            )
        line = (
            f"[effort] |bias|={total_bias:5.1f} "
            f"oob={self.consecutive_out_of_bounds} {worst_txt}  {vals}{pos}"
        )
        self._line_width = max(self._line_width, len(line))
        sys.stdout.write("\r" + line.ljust(self._line_width))
        sys.stdout.flush()

