"""Failure-episode and operator-request conversation state for LOKA orchestration."""

from __future__ import annotations

from loka.agent.directives import format_listeners, format_mission_directives
from loka.agent.plan import format_plan
from loka.agent.robot_context import format_current_model_belief


def _format_context_sections(
    mpc_config: str,
    telemetry_header: str,
    telemetry: str,
    loka_state: dict,
) -> str:
    belief = format_current_model_belief(loka_state)
    return (
        "## LOCKED MISSION DIRECTIVES\n"
        f"{format_mission_directives(loka_state)}\n\n"
        "## SELF-ASSIGNED PLAN\n"
        f"{format_plan(loka_state)}\n\n"
        "## ACTIVE LISTENERS\n"
        f"{format_listeners(loka_state)}\n\n"
        f"{mpc_config}\n\n"
        "## CURRENT MODEL BELIEF\n"
        f"{belief}\n\n"
        f"{telemetry_header}\n"
        f"{telemetry}"
    )


def _format_intervention_history(interventions, empty_label):
    if not interventions:
        return empty_label

    lines = []
    for index, fix in enumerate(interventions, start=1):
        time_label = fix["time"]
        time_str = f"t={time_label:.2f}s" if time_label is not None else "t=unknown"
        lines.append(f"Fix #{index} ({time_str}):")
        lines.append(f"  Hypothesis: {fix.get('hypothesis', 'N/A')}")
        lines.append(f"  Analysis: {fix.get('analysis', 'N/A')}")

        if fix.get("planner_targets"):
            lines.append(f"  Planner_Targets: {fix['planner_targets']}")
        if fix.get("task_targets"):
            lines.append(f"  Task_Targets: {fix['task_targets']}")
        if fix.get("model_mutations"):
            lines.append("  Model_Mutations:")
            for mutation in fix["model_mutations"]:
                lines.append(
                    f"    - {mutation.get('object_type')} '{mutation.get('name')}' "
                    f"{mutation.get('attribute')} = {mutation.get('value')}"
                )
        searches = fix.get("belief_searches") or []
        if isinstance(searches, dict):
            searches = [searches]
        if searches:
            lines.append("  Belief_Searches:")
            for search in searches:
                if isinstance(search, dict):
                    axis = search.get("axis")
                    axis_txt = f" axis={axis}" if axis else ""
                    lines.append(
                        f"    - {search.get('kind') or search.get('attribute')} "
                        f"{search.get('name') or search.get('body') or ''}{axis_txt}"
                    )
                else:
                    lines.append(f"    - {search}")
        for result in fix.get("search_results") or []:
            lines.append(f"  {result}")
        lines.append("")

    return "\n".join(lines).rstrip()


class ConversationMixin:
    def __init__(self):
        self.interventions = []
        self._messages = []

    @property
    def prior_turn_count(self) -> int:
        return len(self._messages) // 2

    def record_intervention(self, sim_time: float, scratchpad: dict) -> None:
        sem_state = scratchpad.get("Semantic_State")
        if not isinstance(sem_state, dict):
            sem_state = {}
        self.interventions.append({
            "time": sim_time,
            "hypothesis": sem_state.get("Hypothesis"),
            "hypotheses": sem_state.get("Hypotheses") or [],
            "analysis": sem_state.get("Analysis"),
            "planner_targets": scratchpad.get("Planner_Targets") or {},
            "task_targets": scratchpad.get("Task_Targets") or {},
            "model_mutations": scratchpad.get("Model_Mutations") or [],
            "belief_searches": scratchpad.get("Belief_Searches")
            or scratchpad.get("Belief_Search")
            or [],
            "search_results": [],
        })

    def append_exchange(self, user_content: str, assistant_content: str) -> None:
        self._messages.append({"role": "user", "content": user_content})
        self._messages.append({"role": "assistant", "content": assistant_content})

    def shows_prior_turns(self, max_memory: int | None) -> bool:
        """True while this conversation is still inside its memory window.

        ``max_memory`` is how many completed turns stay visible. The next turn,
        and every turn after it, is built with no earlier messages and no
        attempted fixes. ``None`` keeps the full history.
        """
        if max_memory is None:
            return True
        return self.prior_turn_count < max_memory

    def compose_api_messages(
        self,
        system_prompt: str,
        user_content: str,
        *,
        max_memory: int | None = None,
        image_png: bytes | None = None,
        image_detail: str = "high",
    ) -> list:
        messages = [{"role": "system", "content": system_prompt}]
        if self.shows_prior_turns(max_memory):
            messages.extend(self._messages)
        messages.append({"role": "user", "content": user_content})
        if image_png is not None:
            attach_image(messages, image_png, detail=image_detail)
        return messages


def attach_image(messages: list, image_png: bytes, *, detail: str = "high") -> None:
    """Turn the last user message into text plus one inline PNG.

    History stays text-only (``append_exchange`` stores the string), so a turn
    only ever carries the frame captured for it.
    """
    import base64

    last = messages[-1]
    text = last["content"] if isinstance(last["content"], str) else _message_text(last)
    url = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
    last["content"] = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": url, "detail": detail}},
    ]


def append_text(messages: list, note: str) -> None:
    """Append a line to the last user message, string or multimodal."""
    last = messages[-1]
    if isinstance(last["content"], str):
        last["content"] = last["content"].rstrip() + "\n\n" + note
        return
    for part in last["content"]:
        if part.get("type") == "text":
            part["text"] = part["text"].rstrip() + "\n\n" + note
            return


def _message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return "\n".join(
        part.get("text", "") for part in content or () if part.get("type") == "text"
    )


_STOOD_FOR_FAULT = (
    "The runtime already set gait.mode=stand and gait.speed=0 before this call. "
    "It does that whenever the effort bias crosses the threshold while the robot "
    "is walking, and it does it before you are invoked. Do not command the stand "
    "yourself. Diagnose the fault while standing, including belief searches if a "
    "model parameter is wrong. The stand is temporary. Do not set gait.mode=walk "
    "on this turn if |bias| is still above the threshold. Stay standing and "
    "queue the search. On a later turn, after |bias| is under the threshold, "
    "set gait.mode=walk, gait.goal_x, gait.goal_y, and gait.speed. "
    "Do not leave the robot standing once the residual is in bounds and the "
    "goal is unfinished."
)

_RESUME_GOAL = (
    "The runtime is holding stand after it stopped a walk for an effort-bias "
    "fault. That stand is not the task. The locked goal is still unfinished and "
    "it is the priority once the fix is done. If ATTEMPTED FIXES shows the belief "
    "searches settled and the effort residual is back in bounds, do not keep "
    "diagnosing and do not stay in stand. This turn, set gait.mode=walk, "
    "gait.goal_x, gait.goal_y, and gait.speed, and keep the mission listeners "
    "armed. If the residual is still open, "
    "finish that fix while standing, then walk on the turn it settles."
)

_RESUME_GOAL_FRESH = (
    "The runtime is holding stand after it stopped a walk for an effort-bias "
    "fault. That stand is not the task. The locked goal is still unfinished and "
    "it is the priority once the fix is done. Earlier turns are not shown. "
    "If the effort residual is back in bounds, do not keep diagnosing and do not "
    "stay in stand. This turn, set gait.mode=walk, gait.goal_x, gait.goal_y, and "
    "gait.speed, and keep the mission listeners armed. If the residual is still "
    "open, finish that fix while standing, then walk on the turn it settles."
)


class FailureEpisode(ConversationMixin):
    """One continuous LLM conversation from first fault until recovery."""

    def __init__(self, started_at: float):
        super().__init__()
        self.started_at = started_at
        self.interrupted_walk = False
        self._announced_stand = False
        self.turns_sent = 0
        self._turn_cap_noted = False

    def _budget_note(self, max_turns: int | None) -> str:
        if not max_turns or max_turns < 1:
            return ""
        number = self.turns_sent + 1
        last = (
            " This is the last turn until the bias is back in bounds."
            if number >= max_turns
            else ""
        )
        return (
            f"Orchestrator turn {number} of {max_turns} this episode. "
            f"The count resets when the bias returns to normal.{last}\n"
        )

    def _gait_note(self, *, just_stood: bool = False, recall: bool = True) -> str:
        if just_stood:
            self.interrupted_walk = True
        if not self.interrupted_walk:
            return ""
        if just_stood or not self._announced_stand:
            self._announced_stand = True
            return f"{_STOOD_FOR_FAULT}\n"
        if recall:
            return f"{_RESUME_GOAL}\n"
        return f"{_RESUME_GOAL_FRESH}\n"

    @property
    def round_number(self) -> int:
        return len(self.interventions) + 1

    def format_attempted_fixes(self, *, max_memory: int | None = None) -> str:
        if not self.shows_prior_turns(max_memory):
            return "None. Earlier turns in this episode are not included."
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
        *,
        just_stood: bool = False,
        max_turns: int | None = None,
        max_memory: int | None = None,
    ) -> str:
        objective = loka_state.get("primary_objective", "")
        recall = self.shows_prior_turns(max_memory)
        walking_note = self._gait_note(just_stood=just_stood, recall=recall)
        if recall:
            situation = (
                "The robot is still unstable after prior intervention(s). "
                "Read ATTEMPTED FIXES. A search named only in Hypothesis did not run.\n"
            )
        else:
            situation = (
                "Earlier orchestrator turns are omitted. Reason from the current "
                "telemetry, tunables, and model belief. Applied gait targets, "
                "search results, and model mutations remain in effect.\n"
            )
        return (
            f"--- FAILURE EPISODE (round {self.round_number}, t={sim_time:.2f}s) ---\n"
            f"Episode started at t={self.started_at:.2f}s. "
            f"{situation}"
            f"{self._budget_note(max_turns)}"
            f"{walking_note}"
            f"Primary objective: {objective}\n\n"
            "## ATTEMPTED FIXES\n"
            f"{self.format_attempted_fixes(max_memory=max_memory)}\n\n"
            + _format_context_sections(
                mpc_config,
                "## NEW TELEMETRY",
                telemetry,
                loka_state,
            )
        )

    def build_initial_user_turn(
        self,
        telemetry: str,
        loka_state: dict,
        sim_time: float,
        mpc_config: str,
        *,
        just_stood: bool = False,
        max_turns: int | None = None,
        max_memory: int | None = None,
    ) -> str:
        objective = loka_state.get("primary_objective", "")
        walking_note = self._gait_note(just_stood=just_stood)
        return (
            f"--- FAILURE EPISODE (round 1, t={sim_time:.2f}s) ---\n"
            "Initial instability detected. Queue Belief_Searches for the model mismatch.\n"
            f"{self._budget_note(max_turns)}"
            f"{walking_note}"
            f"Primary objective: {objective}\n\n"
            "## ATTEMPTED FIXES\n"
            f"{self.format_attempted_fixes(max_memory=max_memory)}\n\n"
            + _format_context_sections(
                mpc_config,
                "## NEW TELEMETRY",
                telemetry,
                loka_state,
            )
        )


class OperatorSession(ConversationMixin):
    """Multi-turn manual operator requests (gait changes, strategy experiments)."""

    def format_prior_interventions(self, *, max_memory: int | None = None) -> str:
        if not self.shows_prior_turns(max_memory):
            return "None. Earlier turns in this session are not included."
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
        *,
        max_memory: int | None = None,
    ) -> str:
        if self.shows_prior_turns(max_memory):
            recall = ""
        else:
            recall = (
                "Earlier orchestrator turns are omitted. Reason from the current "
                "telemetry, tunables, and model belief. Applied gait targets, "
                "search results, and model mutations remain in effect.\n\n"
            )
        return (
            f"--- OPERATOR DIRECTIVE (t={sim_time:.2f}s) ---\n"
            f"{request.strip()}\n\n"
            "This sentence is now a locked mission directive. You cannot change it. "
            "Drive the robot with Task_Targets (including gait.*) and Listeners. "
            "Do not assume hardware failure unless telemetry supports it.\n\n"
            f"{recall}"
            "## PRIOR INTERVENTIONS THIS SESSION\n"
            f"{self.format_prior_interventions(max_memory=max_memory)}\n\n"
            + _format_context_sections(
                mpc_config,
                "## CURRENT TELEMETRY SNAPSHOT",
                telemetry,
                loka_state,
            )
        )

    def build_listener_turn(
        self,
        messages: list[str],
        telemetry: str,
        loka_state: dict,
        sim_time: float,
        mpc_config: str,
    ) -> str:
        body = "\n".join(f"- {m}" for m in messages)
        return (
            f"--- LISTENER (t={sim_time:.2f}s) ---\n"
            "A condition you armed is now true. The telemetry snapshot is current.\n"
            f"{body}\n\n"
            "Act on the message. Keep pursuing every locked directive. "
            "If gait.mode is stand after a fault fix and the goal is unfinished, "
            "set gait.mode=walk, gait.goal_x, and gait.goal_y only when |bias| "
            "is under the threshold. Arm the next listener if the directive is not "
            "finished, or clear listeners with an empty list when it is.\n\n"
            + _format_context_sections(
                mpc_config,
                "## CURRENT TELEMETRY SNAPSHOT",
                telemetry,
                loka_state,
            )
        )

    def build_heartbeat_turn(
        self,
        telemetry: str,
        loka_state: dict,
        sim_time: float,
        mpc_config: str,
        *,
        period_s: float,
    ) -> str:
        return (
            f"--- PERIODIC CHECK (t={sim_time:.2f}s, every {period_s:.0f}s) ---\n"
            "No listener invoked you in this interval. Read closing speed and "
            "along heading in the full-state snapshot. Positive closing speed "
            "is motion toward the goal. Negative means the robot is walking "
            "away; planar speed has no sign. If it is walking away, set "
            "gait.goal_x and gait.goal_y again. If gait.mode is stand because a "
            "fault was fixed, |bias| is under the threshold, and the goal is "
            "unfinished, set gait.mode=walk, gait.goal_x, gait.goal_y, and "
            "gait.speed. Leave listeners armed "
            "if they are still the right wake-up.\n\n"
            + _format_context_sections(
                mpc_config,
                "## CURRENT TELEMETRY SNAPSHOT",
                telemetry,
                loka_state,
            )
        )
