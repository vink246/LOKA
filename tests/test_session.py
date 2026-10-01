"""Cross-session fix window for the LOKA orchestrator (no MuJoCo required)."""

import unittest

from loka.session import VISIBLE_FIX_LIMIT, FailureEpisode, FixWindow, OperatorSession


def _scratchpad(hypothesis, *, gear=None):
    pad = {
        "Semantic_State": {"Hypothesis": hypothesis, "Analysis": hypothesis},
        "Controller_Targets": {},
        "Model_Mutations": [],
    }
    if gear is not None:
        pad["Model_Mutations"] = [
            {
                "object_type": "actuator",
                "name": "right_hip",
                "attribute": "gear",
                "value": gear,
            }
        ]
    return pad


class FixWindowTests(unittest.TestCase):
    def test_default_limit_is_ten(self):
        self.assertEqual(VISIBLE_FIX_LIMIT, 10)
        self.assertEqual(FixWindow().limit, 10)

    def test_window_keeps_last_n_across_sessions(self):
        window = FixWindow(limit=3)
        first = FailureEpisode(1.0, fix_window=window)
        for i in range(3):
            first.record_intervention(float(i + 1), _scratchpad(f"old-{i}"))

        second = FailureEpisode(10.0, fix_window=window)
        second.record_intervention(11.0, _scratchpad("new", gear=0.0))

        hypotheses = [fix["hypothesis"] for fix in window.fixes()]
        self.assertEqual(hypotheses, ["old-1", "old-2", "new"])
        self.assertEqual(len(first.interventions), 3)
        self.assertEqual(second.interventions[0]["session"], "failure episode started at t=10.00s")

        turn = second.build_initial_user_turn("telemetry", {}, 12.0, "mpc")
        self.assertIn("## RECENT FIXES (last 3 across sessions)", turn)
        self.assertIn("old-1", turn)
        self.assertNotIn("old-0", turn)
        self.assertIn("failure episode started at t=1.00s", turn)
        self.assertIn("right_hip", turn)
        self.assertIn("None. This is the first intervention for this failure episode.", turn)

    def test_operator_and_failure_share_the_window(self):
        window = FixWindow(limit=10)
        operator = OperatorSession(fix_window=window)
        operator.record_intervention(2.0, _scratchpad("crouch"))
        episode = FailureEpisode(5.0, fix_window=window)
        turn = episode.build_initial_user_turn("telemetry", {}, 6.0, "mpc")
        self.assertIn("operator session", turn)
        self.assertIn("crouch", turn)

        request = operator.build_request_turn("hop", "telemetry", {}, 7.0, "mpc")
        self.assertIn("## RECENT FIXES (last 10 across sessions)", request)
        self.assertIn("crouch", request)


if __name__ == "__main__":
    unittest.main()
