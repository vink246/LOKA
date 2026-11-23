"""Whole-body optimal control problem over a contact plan (Crocoddyl).

Each knot is a contact-constrained forward-dynamics model of the reduced G1
with rigid 6D sole contacts on the feet the plan has planted. The costs are
the usual ones of whole-body trajectory optimisation with DDP:

- the swing sole follows the plan's swing path and arrives with no velocity;
- each planted sole's wrench stays inside its contact wrench cone (friction
  pyramid, centre of pressure inside the sole);
- joint positions and torques stay inside their limits (quadratic barriers);
- a soft centre-of-mass guide leans the body over the foot that is about to
  carry it, and posture and torque regularisation resolve the rest.

Nothing here prescribes how the body gets its weight from one foot to the
next: the solver finds knee bend, trunk motion and momentum that satisfy the
contacts and the limits, which is what the reduced-order stacks cannot do
when a static transfer does not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import crocoddyl
import numpy as np
import pinocchio as pin

from loka.control.wbto.footsteps import ContactPlan
from loka.control.wbto.model import SOLE_HALF_LENGTH, SOLE_HALF_WIDTH, WholeBodyModel


@dataclass
class OcpParams:
    dt: float = 0.02  # knot spacing [s]
    friction_mu: float = 0.5
    sole_shrink: float = 0.8  # fraction of the sole the centre of pressure may use
    contact_gains: tuple[float, float] = (0.0, 40.0)  # Baumgarte position, velocity
    com_height: float = 0.68  # guide height of the CoM above the support [m]
    com_inset: float = 0.75  # how far toward the stance foot the guide sways (1 = over it)

    w_swing: float = 2.0e4
    w_swing_rot: float = 0.2  # rotation rows, relative to translation
    w_land: float = 1.0e5  # sole at the foothold on the knot it lands
    w_land_vel: float = 2.0e2
    w_com: float = 2.0e2
    w_com_z: float = 0.2  # height row, relative to the horizontal rows
    w_wrench: float = 1.0e0
    w_posture: float = 1.0
    w_torque: float = 1.0e-3  # also sets how stiff the Riccati feedback is
    w_joint_limit: float = 1.0e4
    w_torque_limit: float = 1.0e0
    torque_margin: float = 0.9  # barrier starts at this fraction of each limit
    joint_margin: float = 0.03  # barrier starts this far inside each joint stop [rad]
    w_terminal_com: float = 2.0e3
    w_terminal_posture: float = 10.0

    # Posture weights per tangent coordinate group.
    wq_base_rot: tuple[float, float, float] = (20.0, 20.0, 10.0)
    wq_leg: float = 0.05
    wq_waist: float = 5.0
    wv_base_lin: float = 0.5
    wv_base_ang: float = 2.0
    wv_leg: float = 0.02
    wv_waist: float = 0.5


@dataclass
class Knot:
    """What one knot of the horizon asks of the robot."""

    t: float
    support: dict[int, pin.SE3]  # planted feet and their sole placements
    com: np.ndarray  # CoM guide
    swing: tuple[int, pin.SE3] | None = None  # swinging foot and its sole target
    landed: tuple[int, pin.SE3] | None = None  # foot that touches down on this knot
    step: int = -1  # index of the swing in progress or about to start


def _placement(pos: np.ndarray, yaw: float) -> pin.SE3:
    c, s = np.cos(yaw), np.sin(yaw)
    return pin.SE3(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.asarray(pos, float))


def build_knots(plan: ContactPlan, com0: np.ndarray, params: OcpParams) -> list[Knot]:
    """Knot sequence for the whole plan, ending with the settle after the last swing."""
    p = params
    feet = {f.foot: _placement(f.pos, f.yaw) for f in plan.stance}

    # CoM guide: over (a fraction toward) each stance foot at the middle of its swing.
    mid = np.mean([f.pos for f in plan.stance], axis=0)
    anchors_t, anchors_c = [0.0], [np.array([com0[0], com0[1], mid[2] + p.com_height])]
    t = 0.0
    stance = {f.foot: f for f in plan.stance}
    for step in plan.steps:
        t += step.transfer
        support = stance[1 - step.foot]
        other = 0.5 * (step.start.pos + step.end.pos)
        xy = support.pos[:2] + (1.0 - p.com_inset) * (other[:2] - support.pos[:2])
        anchors_t.append(t + 0.5 * step.duration)
        anchors_c.append(np.array([xy[0], xy[1], support.pos[2] + p.com_height]))
        t += step.duration
        stance[step.foot] = step.end
    final = np.mean([f.pos for f in stance.values()], axis=0)
    anchors_t.append(t + 0.6 * plan.params.settle)
    anchors_c.append(np.array([final[0], final[1], final[2] + p.com_height]))
    anchors_t, anchors_c = np.array(anchors_t), np.array(anchors_c)

    def com_at(time: float) -> np.ndarray:
        return np.array([np.interp(time, anchors_t, anchors_c[:, i]) for i in range(3)])

    knots: list[Knot] = []
    t = 0.0
    landed = None
    for index, step in enumerate(plan.steps):
        for _ in range(max(1, int(round(step.transfer / p.dt)))):
            knots.append(Knot(t, dict(feet), com_at(t), landed=landed, step=index))
            landed = None
            t += p.dt
        n = max(2, int(round(step.duration / p.dt)))
        support = {1 - step.foot: feet[1 - step.foot]}
        for i in range(n):
            tau = i / n
            pos, _ = step.point(tau)
            target = _placement(pos, step.yaw(tau))
            knots.append(Knot(t, support, com_at(t), swing=(step.foot, target), step=index))
            t += p.dt
        feet[step.foot] = _placement(step.end.pos, step.end.yaw)
        landed = (step.foot, feet[step.foot])
    for _ in range(max(1, int(round(plan.params.settle / p.dt)))):
        knots.append(Knot(t, dict(feet), com_at(t), landed=landed, step=len(plan.steps)))
        landed = None
        t += p.dt
    return knots


class OcpBuilder:
    """Action models for knots of a contact plan."""

    def __init__(self, wb: WholeBodyModel, params: OcpParams | None = None) -> None:
        self.wb = wb
        self.p = params or OcpParams()
        self.state = crocoddyl.StateMultibody(wb.model)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        self.nu = self.actuation.nu
        p = self.p
        nj = wb.nv - 6
        n_leg = 12
        self._posture_weights = np.array(
            [0.0, 0.0, 0.0, *p.wq_base_rot]
            + [p.wq_leg] * n_leg
            + [p.wq_waist] * (nj - n_leg)
            + [p.wv_base_lin] * 3
            + [p.wv_base_ang] * 3
            + [p.wv_leg] * n_leg
            + [p.wv_waist] * (nj - n_leg)
        )
        # Joint-range barrier on the tangent of the neutral configuration.
        lb = np.full(self.state.ndx, -np.inf)
        ub = np.full(self.state.ndx, np.inf)
        lb[6 : wb.nv] = wb.model.lowerPositionLimit[7:] + p.joint_margin
        ub[6 : wb.nv] = wb.model.upperPositionLimit[7:] - p.joint_margin
        self._joint_bounds = crocoddyl.ActivationBounds(lb, ub)
        limit = p.torque_margin * wb.torque_limit
        self._torque_bounds = crocoddyl.ActivationBounds(-limit, limit)
        self._gains = np.array(p.contact_gains, dtype=float)
        self._box = 2.0 * p.sole_shrink * np.array([SOLE_HALF_LENGTH, SOLE_HALF_WIDTH])

    # -- models -----------------------------------------------------------

    def posture_reference(self, yaw: float) -> np.ndarray:
        x = self.wb.x0.copy()
        quat = pin.Quaternion(pin.utils.rpyToMatrix(0.0, 0.0, float(yaw)))
        x[3:7] = quat.coeffs()
        return x

    def model(self, knot: Knot, yaw: float = 0.0, terminal: bool = False):
        p, state, nu = self.p, self.state, self.nu
        contacts = crocoddyl.ContactModelMultiple(state, nu)
        for foot, placement in knot.support.items():
            contacts.addContact(
                f"foot{foot}",
                crocoddyl.ContactModel6D(
                    state, self.wb.foot_ids[foot], placement, pin.LOCAL_WORLD_ALIGNED, nu, self._gains
                ),
            )
        costs = crocoddyl.CostModelSum(state, nu)

        posture = crocoddyl.CostModelResidual(
            state,
            crocoddyl.ActivationModelWeightedQuad(self._posture_weights),
            crocoddyl.ResidualModelState(state, self.posture_reference(yaw), nu),
        )
        costs.addCost("posture", posture, p.w_terminal_posture if terminal else p.w_posture)
        costs.addCost(
            "joint_limits",
            crocoddyl.CostModelResidual(
                state,
                crocoddyl.ActivationModelQuadraticBarrier(self._joint_bounds),
                crocoddyl.ResidualModelState(state, state.zero(), nu),
            ),
            p.w_joint_limit,
        )
        costs.addCost(
            "com",
            crocoddyl.CostModelResidual(
                state,
                crocoddyl.ActivationModelWeightedQuad(np.array([1.0, 1.0, p.w_com_z])),
                crocoddyl.ResidualModelCoMPosition(state, knot.com, nu),
            ),
            p.w_terminal_com if terminal else p.w_com,
        )
        if knot.swing is not None:
            foot, target = knot.swing
            costs.addCost(
                "swing",
                crocoddyl.CostModelResidual(
                    state,
                    crocoddyl.ActivationModelWeightedQuad(
                        np.array([1.0, 1.0, 1.0, p.w_swing_rot, p.w_swing_rot, p.w_swing_rot])
                    ),
                    crocoddyl.ResidualModelFramePlacement(state, self.wb.foot_ids[foot], target, nu),
                ),
                p.w_swing,
            )
        if knot.landed is not None:
            foot, target = knot.landed
            costs.addCost(
                "land",
                crocoddyl.CostModelResidual(
                    state,
                    crocoddyl.ResidualModelFramePlacement(state, self.wb.foot_ids[foot], target, nu),
                ),
                p.w_land,
            )
            costs.addCost(
                "land_vel",
                crocoddyl.CostModelResidual(
                    state,
                    crocoddyl.ResidualModelFrameVelocity(
                        state, self.wb.foot_ids[foot], pin.Motion.Zero(), pin.LOCAL_WORLD_ALIGNED, nu
                    ),
                ),
                p.w_land_vel,
            )
        if not terminal:
            costs.addCost(
                "torque",
                crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state, nu)),
                p.w_torque,
            )
            costs.addCost(
                "torque_limits",
                crocoddyl.CostModelResidual(
                    state,
                    crocoddyl.ActivationModelQuadraticBarrier(self._torque_bounds),
                    crocoddyl.ResidualModelControl(state, nu),
                ),
                p.w_torque_limit,
            )
            for foot, placement in knot.support.items():
                cone = crocoddyl.WrenchCone(placement.rotation, p.friction_mu, self._box)
                costs.addCost(
                    f"wrench{foot}",
                    crocoddyl.CostModelResidual(
                        state,
                        crocoddyl.ActivationModelQuadraticBarrier(
                            crocoddyl.ActivationBounds(cone.lb, cone.ub)
                        ),
                        crocoddyl.ResidualModelContactWrenchCone(
                            state, self.wb.foot_ids[foot], cone, nu, True
                        ),
                    ),
                    p.w_wrench,
                )
        dam = crocoddyl.DifferentialActionModelContactFwdDynamics(
            state, self.actuation, contacts, costs, 0.0, True
        )
        return crocoddyl.IntegratedActionModelEuler(dam, 0.0 if terminal else p.dt)

    def problem(self, x0: np.ndarray, knots: list[Knot], yaw: float = 0.0):
        """Shooting problem over ``knots``; the last one is the terminal node."""
        running = [self.model(k, yaw) for k in knots[:-1]]
        terminal = self.model(knots[-1], yaw, terminal=True)
        return crocoddyl.ShootingProblem(x0, running, terminal)


def audit(builder: OcpBuilder, problem, knots: list[Knot], xs, us) -> dict:
    """Physical margins of a trajectory: what the costs only encourage.

    Fractions are of the hard limit (1.0 is at the limit): centre of pressure
    against the full sole, tangential force against ``mu`` times the normal
    force, torque against the actuator limit. ``joint_margin`` is the
    smallest distance to a joint stop [rad].
    """
    wb = builder.wb
    problem.calc(list(xs), list(us))
    cop_len = cop_wid = friction = torque = swing_err = land_speed = 0.0
    fz_min = np.inf
    joint_margin = np.inf
    lower, upper = wb.model.lowerPositionLimit[7:], wb.model.upperPositionLimit[7:]
    data = wb.model.createData()
    for k, knot in enumerate(knots[: problem.T]):
        x, u = np.asarray(xs[k]), np.asarray(us[k])
        torque = max(torque, float(np.max(np.abs(u) / wb.torque_limit)))
        joints = x[7 : wb.nq]
        joint_margin = min(joint_margin, float(np.min(np.minimum(joints - lower, upper - joints))))
        contacts = problem.runningDatas[k].differential.multibody.contacts.contacts
        for foot, placement in knot.support.items():
            wrench = contacts[f"foot{foot}"].f
            rot = placement.rotation
            force, moment = rot.T @ wrench.linear, rot.T @ wrench.angular
            fz = float(force[2])
            fz_min = min(fz_min, fz)
            if fz > 1.0:
                cop_len = max(cop_len, abs(moment[1] / fz) / SOLE_HALF_LENGTH)
                cop_wid = max(cop_wid, abs(moment[0] / fz) / SOLE_HALF_WIDTH)
                friction = max(friction, float(np.hypot(force[0], force[1]) / fz) / builder.p.friction_mu)
        if knot.swing is not None or knot.landed is not None:
            pin.forwardKinematics(wb.model, data, x[: wb.nq], x[wb.nq :])
            pin.updateFramePlacements(wb.model, data)
            foot, target = knot.swing if knot.swing is not None else knot.landed
            sole = data.oMf[wb.foot_ids[foot]]
            swing_err = max(swing_err, float(np.linalg.norm(sole.translation - target.translation)))
            if knot.landed is not None:
                vel = pin.getFrameVelocity(
                    wb.model, data, wb.foot_ids[foot], pin.LOCAL_WORLD_ALIGNED
                )
                land_speed = max(land_speed, float(np.linalg.norm(vel.linear)))
    return {
        "cop_along": round(cop_len, 3),
        "cop_across": round(cop_wid, 3),
        "friction": round(friction, 3),
        "fz_min": round(float(fz_min), 1),
        "torque": round(torque, 3),
        "joint_margin": round(float(joint_margin), 3),
        "swing_error_m": round(swing_err, 4),
        "landing_speed": round(land_speed, 4),
    }


@dataclass
class Solution:
    xs: list[np.ndarray]
    us: list[np.ndarray]
    gains: list[np.ndarray]
    iterations: int
    cost: float
    converged: bool
    stop: float = 0.0
    info: dict = field(default_factory=dict)


def solve(problem, xs=None, us=None, max_iter: int = 100, threads: int = 0,
          verbose: bool = False) -> Solution:
    """FDDP from a quasi-static guess, or from the ``xs``/``us`` given."""
    if threads:
        problem.nthreads = int(threads)
    solver = crocoddyl.SolverFDDP(problem)
    solver.th_stop = 1e-5
    if verbose:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])
    if xs is None:
        xs = [problem.x0.copy() for _ in range(problem.T + 1)]
    if us is None:
        us = problem.quasiStatic(xs[:-1])
    converged = solver.solve(list(xs), list(us), max_iter, False)
    return Solution(
        xs=[np.array(x) for x in solver.xs],
        us=[np.array(u) for u in solver.us],
        gains=[np.array(k) for k in solver.K],
        iterations=int(solver.iter),
        cost=float(solver.cost),
        converged=bool(converged),
        stop=float(solver.stoppingCriteria()),
    )
