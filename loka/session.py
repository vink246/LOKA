"""Failure-episode and operator-request conversation state for LOKA orchestration."""

from __future__ import annotations

from collections import deque

from loka.error_spec import format_error_spec
from loka.robot_context import format_current_model_belief

# How many applied fixes stay in the orchestrator prompt. The window spans
# failure episodes and operator requests; the oldest entry drops when an 11th
# fix is recorded.
VISIBLE_FIX_LIMIT = 10


def _format_context_sections(
    mpc_config: str,
    belief: str,
    telemetry_header: str,
    telemetry: str,
    error_tracking: str | None = None,
    model_parameters: str | None = None,
) -> str:
    parts = [
        "## CURRENT MPC CONFIGURATION\n"
        f"{mpc_config}\n"
    ]
    if error_tracking:
        parts.append(f"\n## CURRENT ERROR TRACKING\n{error_tracking}\n")
    parts.append(f"\n## CURRENT MODEL BELIEF\n{belief}\n")
    if model_parameters:
        parts.append(
            "\n## CURRENT MODEL PARAMETERS (MPC belief)\n"
            f"{model_parameters}\n"
        )
    parts.append(f"\n{telemetry_header}\n{telemetry}")
    return "".join(parts)


def _error_tracking_text(loka_state: dict) -> str:
    error_spec = loka_state.get("error_spec")
    if error_spec is None:
        return "No Error_Tracking configured."
    return format_error_spec(error_spec)


def _format_intervention_history(interventions, empty_label):
    if not interventions:
        return empty_label

    lines = []
    for index, fix in enumerate(interventions, start=1):
        time_label = fix["time"]
        time_str = f"t={time_label:.2f}s" if time_label is not None else "t=unknown"
        session = fix.get("session")
        session_str = f", {session}" if session else ""
        lines.append(f"Fix #{index} ({time_str}{session_str}):")
        lines.append(f"  Hypothesis: {fix.get('hypothesis', 'N/A')}")
        lines.append(f"  Analysis: {fix.get('analysis', 'N/A')}")

        if fix.get("controller_targets"):
            lines.append(f"  Controller_Targets: {fix['controller_targets']}")
        if fix.get("planner_targets"):
            lines.append(f"  Planner_Targets: {fix['planner_targets']}")
        if fix.get("task_targets"):
            lines.append(f"  Task_Targets: {fix['task_targets']}")
        if fix.get("error_tracking"):
            lines.append(f"  Error_Tracking: {fix['error_tracking']}")
        if fix.get("model_mutations"):
            lines.append("  Model_Mutations:")
            for mutation in fix["model_mutations"]:
                lines.append(
                    f"    - {mutation.get('object_type')} '{mutation.get('name')}' "
                    f"{mutation.get('attribute')} = {mutation.get('value')}"
                )
        lines.append("")

    return "\n".join(lines).rstrip()


def _recent_fixes_section(fix_window: FixWindow | None) -> str:
    if fix_window is None:
        return ""
    return (
        f"## RECENT FIXES (last {fix_window.limit} across sessions)\n"
        "Oldest first. This window keeps the last "
        f"{fix_window.limit} fixes from this session and from earlier failure "
        "episodes and operator requests. Fixes older than that are no longer listed.\n"
        f"{fix_window.format()}\n\n"
    )


class FixWindow:
    """Sliding window of applied fixes shared across LOKA sessions."""

    def __init__(self, limit: int = VISIBLE_FIX_LIMIT):
        self.limit = int(limit)
        self._fixes: deque[dict] = deque(maxlen=self.limit)

    def __len__(self) -> int:
        return len(self._fixes)

    def record(self, fix: dict) -> None:
        self._fixes.append(dict(fix))

    def clear(self) -> None:
        self._fixes.clear()

    def fixes(self) -> list[dict]:
        return list(self._fixes)

    def format(self) -> str:
        return _format_intervention_history(self.fixes(), "None yet.")


class ConversationMixin:
    def __init__(self, fix_window: FixWindow | None = None):
        self.interventions = []
        self._messages = []
        self.fix_window = fix_window
        self.session_label: str | None = None

    @property
    def prior_turn_count(self) -> int:
        return len(self._messages) // 2

    def record_intervention(self, sim_time: float, scratchpad: dict) -> None:
        sem_state = scratchpad.get("Semantic_State", {})
        fix = {
            "time": sim_time,
            "hypothesis": sem_state.get("Hypothesis"),
            "analysis": sem_state.get("Analysis"),
            "controller_targets": scratchpad.get("Controller_Targets", {}),
            "planner_targets": scratchpad.get("Planner_Targets", {}),
            "task_targets": scratchpad.get("Task_Targets", {}),
            "error_tracking": scratchpad.get("Error_Tracking"),
            "model_mutations": scratchpad.get("Model_Mutations", []),
        }
        if self.session_label:
            fix["session"] = self.session_label
        self.interventions.append(fix)
        if self.fix_window is not None:
            self.fix_window.record(fix)

    def append_exchange(self, user_content: str, assistant_content: str) -> None:
        self._messages.append({"role": "user", "content": user_content})
        self._messages.append({"role": "assistant", "content": assistant_content})

    def compose_api_messages(self, system_prompt: str, user_content: str) -> list:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._messages)
        messages.append({"role": "user", "content": user_content})
        return messages


class FailureEpisode(ConversationMixin):
    """One continuous LLM conversation from first fault until recovery."""

    def __init__(
        self,
        started_at: float,
        nominal_baseline=None,
        fix_window: FixWindow | None = None,
    ):
        super().__init__(fix_window=fix_window)
        self.started_at = started_at
        self.nominal_baseline = list(nominal_baseline or [])
        self.session_label = f"failure episode started at t={started_at:.2f}s"

    @property
    def round_number(self) -> int:
        return len(self.interventions) + 1

    def format_attempted_fixes(self) -> str:
        return _format_intervention_history(
            self.interventions,
            "None. This is the first intervention for this failure episode.",
        )

    def build_user_turn(
        self,
        telemetry: str,
        loka_state: dict,
        sim_time: float,
        mpc_config: str,
        model_parameters: str | None = None,
    ) -> str:
        belief = format_current_model_belief(loka_state)
        objective = loka_state.get("primary_objective", "")
        return (
            f"--- FAILURE EPISODE (round {self.round_number}, t={sim_time:.2f}s) ---\n"
            f"Episode started at t={self.started_at:.2f}s. "
            "The robot is still unstable after prior intervention(s). "
            "Re-evaluate task parameters and Error_Tracking if the current mission "
            "targets or success criteria are preventing recovery.\n"
            f"Primary objective: {objective}\n\n"
            + _recent_fixes_section(self.fix_window)
            + "## ATTEMPTED FIXES\n"
            f"{self.format_attempted_fixes()}\n\n"
            + _format_context_sections(
                mpc_config,
                belief,
                "## NEW TELEMETRY",
                telemetry,
                error_tracking=_error_tracking_text(loka_state),
                model_parameters=model_parameters,
            )
        )

    def build_initial_user_turn(
        self,
        telemetry: str,
        loka_state: dict,
        sim_time: float,
        mpc_config: str,
        model_parameters: str | None = None,
    ) -> str:
        belief = format_current_model_belief(loka_state)
        objective = loka_state.get("primary_objective", "")
        return (
            f"--- FAILURE EPISODE (round 1, t={sim_time:.2f}s) ---\n"
            "Initial instability detected. Consider adjusting cost weights, planner "
            "settings, task parameters, and Error_Tracking if the current mission "
            "targets are unrealistic for recovery.\n"
            f"Primary objective: {objective}\n\n"
            + _recent_fixes_section(self.fix_window)
            + "## ATTEMPTED FIXES\n"
            "None. This is the first intervention for this failure episode.\n\n"
            + _format_context_sections(
                mpc_config,
                belief,
                "## NEW TELEMETRY",
                telemetry,
                error_tracking=_error_tracking_text(loka_state),
                model_parameters=model_parameters,
            )
        )


class OperatorSession(ConversationMixin):
    """Multi-turn manual operator requests (gait changes, strategy experiments)."""

    def __init__(self, fix_window: FixWindow | None = None):
        super().__init__(fix_window=fix_window)
        self.session_label = "operator session"

    def format_prior_interventions(self) -> str:
        return _format_intervention_history(
            self.interventions,
            "None yet this session.",
        )

    def build_request_turn(
        self,
        request: str,
        telemetry: str,
        loka_state: dict,
        sim_time: float,
        mpc_config: str,
        model_parameters: str | None = None,
    ) -> str:
        belief = format_current_model_belief(loka_state)
        return (
            f"--- OPERATOR REQUEST (t={sim_time:.2f}s) ---\n"
            f"{request.strip()}\n\n"
            "The operator is requesting a deliberate strategy or objective change. "
            "Adjust MPC cost weights, planner metaparameters, task parameters, and "
            "internal model beliefs to explore this request. "
            "Because the overarching objective is changing, you MUST include an "
            "updated Error_Tracking block so success/failure criteria match the new "
            "mission. Do not assume hardware failure unless telemetry supports it.\n\n"
            + _recent_fixes_section(self.fix_window)
            + "## PRIOR INTERVENTIONS THIS SESSION\n"
            f"{self.format_prior_interventions()}\n\n"
            + _format_context_sections(
                mpc_config,
                belief,
                "## CURRENT TELEMETRY SNAPSHOT",
                telemetry,
                error_tracking=_error_tracking_text(loka_state),
                model_parameters=model_parameters,
            )
        )
