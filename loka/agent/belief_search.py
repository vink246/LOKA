"""Deterministic belief-parameter search, driven by the effort residual.

Each model mutation has a search. The control loop calls :meth:`BeliefSearchRunner.tick`
after the residual is updated. The orchestrator queues searches; it does not
pick the numeric value.

The residual is plant bias minus belief bias, resolved into actuators. Its sign
says which way the live belief is wrong:

- mass: mean of the two knee residuals. Positive means the body is too light.
- com x: pitch residual. Positive means the COM is too far back.
- com y: waist-roll residual. Positive means the COM is too far right.
- com z: pitch residual, much weaker than x. Positive means the COM is too low.
- friction and gear do not enter this residual. Their searches still run, and
  stop if a probe leaves the residual unchanged.

Unbounded binary search: while the sign stays the same, double the last step
(mass, friction, and gear multiply; a COM axis doubles its offset from the
nominal coordinate). When the sign flips, bisect. Stop when the signed residual
is inside tolerance, the bracket is tighter than the value tolerance, or a
probe does not move the residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from loka.agent.interactive import MASS_BODY_ALIASES
from loka.agent.model_state import resolve_mjcf_name
from loka.control.gait import MODE_WALK

_KNEE = ("left_knee", "right_knee")
_PITCH = ("left_hip_pitch", "right_hip_pitch", "waist_pitch")


@dataclass(frozen=True)
class SearchRequest:
    kind: str
    name: str
    axis: str | None = None


@dataclass(frozen=True)
class SearchDecision:
    done: bool
    value: float
    reason: str
    error: float
    bias: float


@dataclass
class UnboundedSearch:
    """One-dimensional bracket search. Positive error means the value is too small."""

    start: float
    mode: str
    expand: float = 2.0
    initial_step: float = 0.04
    origin: float = 0.0
    error_tol: float = 0.4
    value_tol: float = 0.05
    unchanged_tol: float = 0.2
    max_probes: int = 14
    lo: float = 0.05
    hi: float = 80.0
    value: float = field(init=False)
    low: float | None = field(default=None, init=False)
    high: float | None = field(default=None, init=False)
    probes: list[tuple[float, float, float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.value = float(self.start)
        self.origin = float(self.origin)

    def observe(self, error: float, bias: float) -> SearchDecision:
        error = float(error)
        bias = float(bias)
        self.probes.append((self.value, error, bias))
        if abs(error) <= self.error_tol:
            return SearchDecision(True, self.value, "bias settled", error, bias)
        if len(self.probes) >= self.max_probes:
            best = min(self.probes, key=lambda item: abs(item[1]))
            return SearchDecision(True, best[0], "probe limit", best[1], best[2])
        if len(self.probes) >= 2:
            prev_value, prev_error, _prev_bias = self.probes[-2]
            moved = abs(self.value - prev_value) > 0.0
            flat = abs(error - prev_error) <= self.unchanged_tol
            sign_flipped = prev_error * error < 0.0
            if moved and flat and not sign_flipped and (self.low is None or self.high is None):
                return SearchDecision(
                    True, self.start, "bias did not move", error, bias
                )
        if error > 0.0:
            if self.low is None or self.value > self.low:
                self.low = self.value
        elif error < 0.0:
            if self.high is None or self.value < self.high:
                self.high = self.value
        if (
            self.low is not None
            and self.high is not None
            and abs(self.high - self.low) <= self.value_tol
        ):
            return SearchDecision(
                True, 0.5 * (self.low + self.high), "bracket closed", error, bias
            )
        nxt = self._propose()
        if abs(nxt - self.value) <= 1e-9:
            return SearchDecision(True, self.value, "range limit", error, bias)
        self.value = nxt
        return SearchDecision(False, nxt, "probe", error, bias)

    def _propose(self) -> float:
        if self.low is not None and self.high is not None:
            if abs(self.high - self.low) <= self.value_tol:
                return self.value
            return self._clamp(0.5 * (self.low + self.high))
        if self.low is not None:
            return self._clamp(self._expand(+1))
        if self.high is not None:
            return self._clamp(self._expand(-1))
        return self.value

    def _expand(self, direction: int) -> float:
        if self.mode == "add":
            offset = abs(self.value - self.origin)
            offset = self.initial_step if offset < 1e-9 else offset * 2.0
            return self.origin + direction * offset
        factor = self.expand if direction > 0 else 1.0 / self.expand
        return self.value * factor

    def _clamp(self, value: float) -> float:
        return float(min(self.hi, max(self.lo, value)))


def _mean(bias, names, keys: tuple[str, ...]) -> float | None:
    vals = []
    for key in keys:
        if key in names:
            vals.append(float(bias[names.index(key)]))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def signed_error(kind: str, axis: str | None, bias, names) -> float:
    """Positive when the belief parameter should increase."""
    bias = np.asarray(bias, dtype=float)
    if kind == "mass":
        knees = _mean(bias, names, _KNEE)
        return 0.0 if knees is None else knees
    if kind == "com":
        if axis == "y":
            roll = _mean(bias, names, ("waist_roll",))
            return 0.0 if roll is None else roll
        pitch = _mean(bias, names, _PITCH)
        # +x and +z both make the pitch residual positive, so the parameter is too high.
        return 0.0 if pitch is None else -pitch
    if kind in ("friction", "gear"):
        knees = _mean(bias, names, _KNEE)
        return 0.0 if knees is None else knees
    return 0.0


def make_search(kind: str, start: float, *, axis: str | None = None, origin: float | None = None) -> UnboundedSearch:
    origin = start if origin is None else origin
    if kind == "mass":
        return UnboundedSearch(
            start, mode="mul", expand=2.0, origin=origin,
            error_tol=0.4, value_tol=max(0.05, 0.01 * abs(start)),
            lo=0.05, hi=80.0,
        )
    if kind == "com":
        tol = 0.05 if axis == "z" else 0.4
        return UnboundedSearch(
            start, mode="add", initial_step=0.04, origin=origin,
            error_tol=tol, value_tol=0.005,
            lo=origin - 0.4, hi=origin + 0.4,
        )
    if kind == "friction":
        return UnboundedSearch(
            start, mode="mul", expand=3.0, origin=origin,
            error_tol=0.4, value_tol=max(0.02, 0.05 * abs(start)),
            lo=0.02, hi=5.0,
        )
    if kind == "gear":
        return UnboundedSearch(
            start, mode="mul", expand=2.0, origin=origin,
            error_tol=0.4, value_tol=max(0.02, 0.05 * abs(start)),
            lo=0.0, hi=max(5.0, abs(start) * 4.0),
        )
    raise ValueError(f"unknown search kind {kind!r}")


def search_for_keycode(keycode: int) -> SearchRequest | None:
    """Viewer keys: M torso mass, X/Y/Z torso COM on that axis."""
    if not 0 < keycode < 128:
        return None
    letter = chr(keycode).lower()
    if letter == "m":
        return SearchRequest("mass", "torso_link")
    if letter in {"x", "y", "z"}:
        return SearchRequest("com", "torso_link", letter)
    return None


def parse_belief_searches(scratchpad: dict) -> list[SearchRequest]:
    """Read a ``Belief_Searches`` list from an orchestrator scratchpad.

    A COM entry may name one axis or a list of axes. Each axis becomes its own
    search, in the order written.
    """
    raw = scratchpad.get("Belief_Searches")
    if raw is None:
        raw = scratchpad.get("Belief_Search")
    if raw is None:
        return []
    if isinstance(raw, dict):
        items = [raw]
    elif isinstance(raw, list):
        items = raw
    else:
        print("[search] Belief_Searches must be a list")
        return []
    requests: list[SearchRequest] = []
    for item in items:
        requests.extend(_requests_from_item(item))
    return requests


def _body_name(raw: str | None, default: str) -> str:
    if not raw:
        return default
    return MASS_BODY_ALIASES.get(str(raw).lower(), str(raw))


def _axes(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip().lower() for part in value.replace(",", " ").split() if part.strip()]
    elif isinstance(value, (list, tuple)):
        parts = [str(part).strip().lower() for part in value]
    else:
        parts = [str(value).strip().lower()]
    return [part for part in parts if part in {"x", "y", "z"}]


def _requests_from_item(item) -> list[SearchRequest]:
    if not isinstance(item, dict):
        print(f"[search] ignored search entry {item!r}")
        return []
    kind = str(item.get("kind") or item.get("attribute") or item.get("param") or "").strip().lower()
    if kind in {"mu", "friction_mu"}:
        kind = "friction"
    name = item.get("name") or item.get("body") or item.get("geom") or item.get("actuator")
    if kind == "mass":
        return [SearchRequest("mass", _body_name(name, "torso_link"))]
    if kind == "com":
        axes = _axes(item.get("axis", item.get("axes")))
        if not axes:
            print("[search] com search needs axis x, y, or z")
            return []
        body = _body_name(name, "torso_link")
        return [SearchRequest("com", body, axis) for axis in axes]
    if kind == "friction":
        return [SearchRequest("friction", str(name or "floor"))]
    if kind == "gear":
        if not name:
            print("[search] gear search needs an actuator name")
            return []
        return [SearchRequest("gear", str(name))]
    print(f"[search] unknown search kind {kind!r}")
    return []


def parse_search_command(line: str) -> SearchRequest | str | None:
    """Parse ``search mass|com|friction|gear ...`` or ``search stop``.

    Returns None when the line is not a search command.
    """
    parts = line.strip().split()
    if not parts or parts[0].lower() != "search":
        return None
    if len(parts) == 1 or parts[1].lower() in {"stop", "cancel", "off"}:
        return "stop"
    kind = parts[1].lower()
    rest = parts[2:]
    if kind in {"mass", "m"}:
        body = "torso_link"
        if rest:
            body = MASS_BODY_ALIASES.get(rest[0].lower(), rest[0])
        return SearchRequest("mass", body)
    if kind == "com":
        if not rest or rest[0].lower() not in {"x", "y", "z"}:
            print("[search] usage: search com <x|y|z> [body]")
            return None
        axis = rest[0].lower()
        body = "torso_link"
        if len(rest) > 1:
            body = MASS_BODY_ALIASES.get(rest[1].lower(), rest[1])
        return SearchRequest("com", body, axis)
    if kind in {"friction", "mu"}:
        geom = rest[0] if rest else "floor"
        return SearchRequest("friction", geom)
    if kind == "gear":
        if not rest:
            print("[search] usage: search gear <actuator>")
            return None
        return SearchRequest("gear", rest[0])
    print("[search] kinds: mass, com <x|y|z>, friction, gear, stop")
    return None


class BeliefSearchRunner:
    """Applies probes on the belief model and waits for the residual to settle."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime
        self.active = False
        self._pending: SearchRequest | str | None = None
        self._queue: list[SearchRequest] = []
        self._search: UnboundedSearch | None = None
        self._request: SearchRequest | None = None
        self._ready_at = 0.0
        # self._settle_s = 0.4
        self._saved_mutations: list | None = None
        self._target: dict | None = None
        self._saved_gait: dict | None = None
        self._hold_for_stand = False

    @property
    def busy(self) -> bool:
        """True while a search is running or waiting in the queue."""
        return self.active or bool(self._queue) or self._pending is not None

    def request(self, spec: SearchRequest | str) -> None:
        """Start one search, or ``\"stop\"``. The control thread applies it.

        A keyboard request replaces any queue. Orchestrator calls use
        :meth:`enqueue` so several searches survive.
        """
        self._pending = spec

    def enqueue(self, specs: list[SearchRequest]) -> None:
        """Append searches and start the first one if nothing is running."""
        if not specs:
            return
        self._queue.extend(specs)
        labels = ", ".join(_request_label(spec) for spec in specs)
        print(f"\n[search] queued {len(specs)}: {labels}")
        if not self.active:
            self._start_next()

    def tick(self) -> None:
        pending = self._pending
        if pending is not None:
            self._pending = None
            if pending == "stop":
                self._cancel()
            elif isinstance(pending, SearchRequest):
                self._queue.clear()
                self._begin(pending)
        if not self.active or self._search is None or self._target is None:
            return
        now = float(self.runtime.sim.data.time)
        if self._hold_for_stand:
            if self.runtime.sim.fell:
                self._hold_for_stand = False
                self._queue.clear()
                self._finish("robot fell", self._search.value)
                return
            if self.runtime.effort_observer.is_walking:
                return
            self._hold_for_stand = False
            settle = self._get_settle_time()
            self._ready_at = now + settle
            print(
                f"\n[search] {self._label()} standing."
                f" Settling {settle:.2f}s before the first probe."
            )
            return
        if now < self._ready_at:
            return
        if self.runtime.sim.fell:
            self._queue.clear()
            self._finish("robot fell", self._search.value)
            return
        obs = self.runtime.effort_observer
        if obs.joint_bias is None:
            return
        bias = np.asarray(obs.joint_bias, dtype=float)
        total = float(np.sum(np.abs(bias)))
        names = list(obs.names or ())
        error = signed_error(self._request.kind, self._request.axis, bias, names)
        decision = self._search.observe(error, total)
        label = self._label()
        print(
            f"\n[search] {label}  value={self._search.probes[-1][0]:.4g}"
            f"  |bias|={total:.2f}  error={error:+.2f}"
            + ("" if decision.done else f"  next={decision.value:.4g}")
        )
        if decision.done:
            if decision.reason == "bias did not move":
                self._restore()
                self._restore_gait()
                print(
                    f"[search] {label} stopped: effort bias did not move. "
                    f"Restored the previous belief."
                )
                self._record(self._search.start, decision.reason)
                self._search = None
                self.active = False
                self._hold_for_stand = False
                self._start_next()
                return
            self._write(decision.value)
            self._finish(decision.reason, decision.value)
            return
        self._write(decision.value)
        self._ready_at = now + self._get_settle_time() # self._settle_s

    def _get_settle_time(self) -> float:
        obs = self.runtime.effort_observer
        # Calculate how long it takes to completely flush the observer's rolling window
        flush_time = 2.0 * obs.step_period if obs.is_walking else (obs.window_size * obs.dt)
        # Add 0.2s for the physical robot state (PD loops, contacts) to stabilize under the new belief
        return flush_time + 0.2
    
    def _restore_gait(self) -> None:
        if self._saved_gait is None:
            return
        self.runtime.controller.set_task_targets(self._saved_gait)
        print("\n[search] Restored nominal gait.")
        self._saved_gait = None

    def _begin(self, spec: SearchRequest) -> None:
        target = self._resolve(spec)
        if target is None:
            return
        self._saved_mutations = [dict(item) for item in self.runtime.loka_state.get("mutations", [])]
        self._request = spec
        self._target = target
        start = float(target["start"])
        origin = float(target["origin"])
        self._search = make_search(spec.kind, start, axis=spec.axis, origin=origin)
        self.active = True
        now = float(self.runtime.sim.data.time)
        self._engage_measurement_gait()
        if self._hold_for_stand:
            self._ready_at = now
            print(
                f"\n[search] {self._label()} started at {start:.4g} "
                f"(nominal {origin:.4g}). Waiting for the stand before probing."
            )
            return
        settle = self._get_settle_time()
        self._ready_at = now + settle
        print(
            f"\n[search] {self._label()} started at {start:.4g} "
            f"(nominal {origin:.4g}). Settling {settle:.2f}s between probes."
        )

    def _engage_measurement_gait(self) -> None:
        """Keep a fault stand. A live walk may still use the slow search gait.

        ``gait.mode=stand`` only requests a stop. Until that stop finishes,
        ``is_walking`` is still true. Applying the speed-0 search gait there
        widens the stance the stop is trying to square and leaves the step
        clock running, which walks backward under a torso-mass error.
        """
        gait = self.runtime.controller.gait
        if gait.stop_in_progress or float(gait.config.mode) < MODE_WALK:
            self._saved_gait = None
            self._hold_for_stand = bool(gait.walking)
            return
        self._hold_for_stand = False
        if not self.runtime.effort_observer.is_walking:
            self._saved_gait = None
            return
        self._saved_gait = {
            "gait.speed": float(gait.config.speed),
            "gait.stance_width": float(gait.config.stance_width),
            "gait.duty_factor": float(gait.config.duty_factor),
        }
        self.runtime.controller.set_task_targets({
            "gait.speed": 0,
            "gait.stance_width": 0.28,
            "gait.duty_factor": 0.85,
        })
        print("\n[search] Engaged high-stability Search Gait.")

    def _resolve(self, spec: SearchRequest) -> dict | None:
        import mujoco

        model = self.runtime.controller.robot.model
        state = self.runtime.loka_state
        if spec.kind in {"mass", "com"}:
            name, body_id = resolve_mjcf_name(model, mujoco.mjtObj.mjOBJ_BODY, spec.name)
            if body_id < 0:
                print(f"[search] no body named {spec.name!r}")
                return None
            nominal_mass = np.asarray(state["nominal_mass"], dtype=float)
            nominal_ipos = np.asarray(state["nominal_ipos"], dtype=float)
            if spec.kind == "mass":
                start = float(nominal_mass[body_id])
                for mut in state.get("mutations", []):
                    if mut.get("type") == "body" and mut.get("attr") == "mass" and mut.get("name") == name:
                        start = float(mut["val"])
                return {"name": name, "id": body_id, "start": start, "origin": float(nominal_mass[body_id])}
            axis = {"x": 0, "y": 1, "z": 2}[spec.axis or "x"]
            ipos = nominal_ipos[body_id].copy()
            for mut in state.get("mutations", []):
                if mut.get("type") == "body" and mut.get("attr") == "com" and mut.get("name") == name:
                    ipos = np.asarray(mut["val"], dtype=float).reshape(3).copy()
            return {
                "name": name,
                "id": body_id,
                "axis": axis,
                "ipos": ipos,
                "start": float(ipos[axis]),
                "origin": float(nominal_ipos[body_id, axis]),
            }
        if spec.kind == "friction":
            name, geom_id = resolve_mjcf_name(model, mujoco.mjtObj.mjOBJ_GEOM, spec.name)
            if geom_id < 0:
                print(f"[search] no geom named {spec.name!r}")
                return None
            nominal = np.asarray(state["nominal_friction"], dtype=float)
            start = float(nominal[geom_id, 0])
            for mut in state.get("mutations", []):
                if mut.get("type") == "geom" and mut.get("attr") == "friction" and mut.get("name") == name:
                    val = mut["val"]
                    start = float(val[0] if isinstance(val, (list, tuple)) else val)
            return {"name": name, "id": geom_id, "start": start, "origin": float(nominal[geom_id, 0])}
        if spec.kind == "gear":
            name, act_id = resolve_mjcf_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, spec.name)
            if act_id < 0:
                print(f"[search] no actuator named {spec.name!r}")
                return None
            nominal = np.asarray(state["nominal_gears"], dtype=float)
            start = float(nominal[act_id])
            for mut in state.get("mutations", []):
                if mut.get("type") == "actuator" and mut.get("attr") == "gear" and mut.get("name") == name:
                    start = float(mut["val"])
            return {"name": name, "id": act_id, "start": start, "origin": float(nominal[act_id])}
        print(f"[search] unknown kind {spec.kind!r}")
        return None

    def _write(self, value: float) -> None:
        spec = self._request
        target = self._target
        assert spec is not None and target is not None
        if spec.kind == "com":
            ipos = np.asarray(target["ipos"], dtype=float).copy()
            ipos[int(target["axis"])] = float(value)
            target["ipos"] = ipos
            stored = ipos.tolist()
            attr = "com"
            mtype = "body"
        elif spec.kind == "mass":
            stored = float(value)
            attr = "mass"
            mtype = "body"
        elif spec.kind == "friction":
            stored = float(value)
            attr = "friction"
            mtype = "geom"
        else:
            stored = float(value)
            attr = "gear"
            mtype = "actuator"
        _upsert(
            self.runtime.loka_state,
            mtype=mtype,
            obj_id=int(target["id"]),
            attr=attr,
            val=stored,
            name=str(target["name"]),
            applied_at=float(self.runtime.sim.data.time),
        )

    def _restore(self) -> None:
        if self._saved_mutations is not None:
            self.runtime.loka_state["mutations"] = self._saved_mutations

    def _start_next(self) -> None:
        while self._queue:
            spec = self._queue.pop(0)
            self._begin(spec)
            if self.active:
                return
        self.active = False

    def _cancel(self) -> None:
        self._queue.clear()
        if not self.active:
            print("[search] nothing running")
            return
        self._restore()
        self._restore_gait()
        print("[search] cancelled; belief restored")
        self.active = False
        self._hold_for_stand = False
        self._search = None

    def _record(self, value: float, reason: str) -> None:
        line = f"{self._label()} done value={value:.4g} ({reason})"
        note = getattr(self.runtime, "note_search_result", None)
        if note is not None:
            note(line)

    def _finish(self, reason: str, value: float) -> None:
        print(f"[search] {self._label()} done  value={value:.4g}  ({reason})")
        self._record(value, reason)
        self._restore_gait()
        self._search = None
        self.active = False
        self._hold_for_stand = False
        self._start_next()

    def _label(self) -> str:
        spec = self._request
        if spec is None:
            return "search"
        if spec.kind == "com":
            return f"com {spec.axis} {spec.name}"
        return f"{spec.kind} {spec.name}"


def _request_label(spec: SearchRequest) -> str:
    if spec.kind == "com":
        return f"com {spec.axis} {spec.name}"
    return f"{spec.kind} {spec.name}"


def _upsert(loka_state, *, mtype, obj_id, attr, val, name, applied_at) -> None:
    mutations = loka_state.setdefault("mutations", [])
    for mut in mutations:
        if mut.get("type") == mtype and mut.get("name") == name and mut.get("attr") == attr:
            mut["id"] = obj_id
            mut["val"] = val
            mut["applied_at"] = applied_at
            return
    mutations.append(
        {
            "type": mtype,
            "id": obj_id,
            "attr": attr,
            "val": val,
            "name": name,
            "applied_at": applied_at,
        }
    )
