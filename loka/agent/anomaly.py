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
    """Model residual in actuator torques.

    ``plant_bias`` is the measured generalized force: actuator torque (current),
    contact force (foot sensors), and any known external force. ``belief_bias``
    is the belief model's inverse dynamics at the measured position, velocity,
    and acceleration. Their difference is zero when the belief explains the
    motion. Torso mass and COM land on the floating base, so the base wrench is
    resolved through the sole Jacobian, ``J_base.T @ f = Δ_base``, and the
    actuator value is ``J_joint.T @ f`` plus the joint residual.
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


def motor_deficit(model, belief_gear, nominal_gear, ctrl, qfrc_actuator):
    """Commanded torque minus the torque the motors applied, per actuator.

    ``qfrc_actuator`` is the sim stand-in for current-derived torque. Expected
    torque scales with the believed gear, so a gear search can drive this to
    zero. ``gear_error`` is ``(belief_gear / nominal - plant_gear)`` on
    actuators whose command is large enough to divide; positive means the
    belief gear is high.
    """
    ctrl = np.asarray(ctrl, dtype=float).reshape(-1)
    applied = np.asarray(qfrc_actuator, dtype=float).reshape(-1)
    belief = np.asarray(belief_gear, dtype=float).reshape(-1)
    nominal = np.asarray(nominal_gear, dtype=float).reshape(-1)
    deficit = np.zeros(model.nu, dtype=float)
    gear_error = np.zeros(model.nu, dtype=float)
    for i in range(model.nu):
        joint_id = int(model.actuator_trnid[i, 0])
        if joint_id < 0:
            continue
        dof = int(model.jnt_dofadr[joint_id])
        if dof < 0 or dof >= applied.size:
            continue
        base = float(nominal[i]) if abs(float(nominal[i])) > 1e-9 else 1.0
        expected = (float(belief[i]) / base) * float(ctrl[i])
        deficit[i] = expected - float(applied[dof])
        if abs(float(ctrl[i])) > 1.0:
            gear_error[i] = deficit[i] / float(ctrl[i])
    return deficit, gear_error


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
        # |commanded - applied|. Healthy motors sit at 0. A standing knee at
        # 75% gear is about 4 Nm, so 1.5 Nm is above noise and below that fault.
        self.motor_threshold_nm = float(bias.get("motor_threshold_nm", 1.5))
        # Planted-sole tangential speed. A healthy walk averages under 0.11 m/s.
        self.slip_threshold_mps = float(bias.get("slip_threshold_mps", 0.25))
        self.clear_below_s = float(bias.get("clear_below_s", 0.5))
        self.consecutive_out_of_bounds = 0
        self.consecutive_in_bounds = 0
        self.joint_bias = None
        self.motor_bias = None
        self.gear_error = None
        self.motor_nm = 0.0
        self.slip_mps = 0.0
        self.out_of_bounds = False
        self.avg_cost = None
        self._walk = deque()
        self._walk_motor = deque()
        self._walk_gear = deque()
        self._walk_slip = deque()
        self._motor_samples = deque(maxlen=window_size_steps)
        self._gear_samples = deque(maxlen=window_size_steps)
        self._slip_samples = deque(maxlen=window_size_steps)
        self._was_walking = False
        self._line_width = 0
        self._named = False
        self.xyz = None

    def update(self, delta_tau, cost, printout=False, xyz=None, motor=None, slip=0.0, gear_error=None):
        # Signed actuator image of plant bias minus belief bias.
        delta_tau = np.asarray(delta_tau, dtype=float)
        self.torque_deviations.append(delta_tau)
        self.costs.append(np.asarray(cost, dtype=float))
        if motor is None:
            motor = np.zeros(delta_tau.shape, dtype=float)
        motor = np.asarray(motor, dtype=float)
        if gear_error is None:
            gear_error = np.zeros(motor.shape, dtype=float)
        gear_error = np.asarray(gear_error, dtype=float)
        self._motor_samples.append(motor)
        self._gear_samples.append(gear_error)
        self._slip_samples.append(float(slip))

        if self.is_walking:
            # Mean over two step periods. Uniform dt makes the mean the integral mean.
            n = max(1, int(round(2.0 * self.step_period / self.dt)))
            joint_bias = self._push_walk(self._walk, delta_tau, n)
            motor_bias = self._push_walk(self._walk_motor, motor, n)
            gear_bias = self._push_walk(self._walk_gear, gear_error, n)
            slip_m = float(self._push_walk(self._walk_slip, float(slip), n))
        else:
            self._walk.clear()
            self._walk_motor.clear()
            self._walk_gear.clear()
            self._walk_slip.clear()
            joint_bias = np.mean(np.stack(self.torque_deviations), axis=0)
            motor_bias = np.mean(np.stack(self._motor_samples), axis=0)
            gear_bias = np.mean(np.stack(self._gear_samples), axis=0)
            slip_m = float(np.mean(self._slip_samples))
        avg_cost = np.mean(np.stack(self.costs), axis=0)
        if self.is_walking:
            ready = len(self._walk) >= n
        else:
            ready = len(self.torque_deviations) >= self.window_size
        if self.is_walking != self._was_walking:
            self.consecutive_out_of_bounds = 0
            self.consecutive_in_bounds = 0
            self._was_walking = self.is_walking
        # A one-sample mean is the startup spike. Publish zeros, and do not
        # count the step, until the standing window or the two-step walk
        # window is full.
        if not ready:
            joint_bias = np.zeros_like(delta_tau)
            motor_bias = np.zeros_like(motor)
            slip_m = 0.0
            total_bias = 0.0
            motor_nm = 0.0
            self.out_of_bounds = False
        else:
            total_bias = float(np.sum(np.abs(joint_bias)))
            motor_nm = float(np.sum(np.abs(motor_bias)))
            threshold = self.walk_threshold_nm if self.is_walking else self.stand_threshold_nm
            self.out_of_bounds = (
                total_bias > threshold
                or motor_nm > self.motor_threshold_nm
                or slip_m > self.slip_threshold_mps
            )
            if self.out_of_bounds:
                self.consecutive_out_of_bounds += 1
                self.consecutive_in_bounds = 0
            else:
                self.consecutive_out_of_bounds = 0
                self.consecutive_in_bounds += 1
        self.joint_bias = joint_bias
        self.motor_bias = motor_bias
        self.gear_error = gear_bias
        self.motor_nm = motor_nm
        self.slip_mps = slip_m
        self.avg_cost = avg_cost
        if xyz is not None:
            self.xyz = np.asarray(xyz, dtype=float).reshape(3)

        if printout:
            self._write(avg_cost, joint_bias, total_bias)

        return total_bias, joint_bias, avg_cost

    @staticmethod
    def _push_walk(buf: deque, sample, n: int):
        buf.append(sample)
        while len(buf) > n:
            buf.popleft()
        return np.mean(np.stack(buf), axis=0)

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
        motor_txt = ""
        if self.motor_bias is not None and self.motor_bias.size:
            slot = int(np.argmax(np.abs(self.motor_bias)))
            motor_name = names[slot] if slot < len(names) else str(slot)
            motor_txt = (
                f" motor={self.motor_nm:4.1f}({motor_name}={float(self.motor_bias[slot]):+.1f})"
            )
        line = (
            f"[effort] |bias|={total_bias:5.1f}{motor_txt}"
            f" slip={self.slip_mps:4.2f} "
            f"oob={self.consecutive_out_of_bounds} {worst_txt}  {vals}{pos}"
        )
        self._line_width = max(self._line_width, len(line))
        sys.stdout.write("\r" + line.ljust(self._line_width))
        sys.stdout.flush()

