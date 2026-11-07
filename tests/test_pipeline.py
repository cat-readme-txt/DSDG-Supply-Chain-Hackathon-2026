"""Contract checks for generated submissions and the Phase 2 forecast boundary."""

import copy
import tempfile
import unittest
from pathlib import Path

from pipeline import CompetitionData, PROFILES


class PipelineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = CompetitionData(Path(__file__).resolve().parents[1] / "data")

    def test_primary_round_trip_and_two_independent_windows(self) -> None:
        rows = self.data.candidate_rows(PROFILES["primary"])
        quantities = {row["id"]: int(row["order_units"]) for row in rows}
        for material in self.data.materials:
            self.assertGreater(quantities[f"w209_{material}"], 0)
            self.assertEqual(quantities[f"w217_{material}"], 0)
            self.assertGreater(quantities[f"w222_{material}"], 0)
            self.assertEqual(quantities[f"w230_{material}"], 0)
        with tempfile.TemporaryDirectory() as folder:
            output = self.data.write_candidate(PROFILES["primary"], Path(folder))
            self.data.validate_file(output)

    def test_rejects_ineligible_share_and_duplicate_id(self) -> None:
        rows = self.data.candidate_rows(PROFILES["primary"])
        invalid = copy.deepcopy(rows)
        invalid[0]["share_AUR"] = "0.1"
        invalid[0]["share_VAL"] = "0.233333"
        invalid[0]["share_ORI"] = "0.233333"
        invalid[0]["share_KES"] = "0.433334"
        with self.assertRaisesRegex(ValueError, "ineligible country"):
            self.data.validate_rows(invalid)
        invalid = copy.deepcopy(rows)
        invalid[1]["id"] = invalid[0]["id"]
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            self.data.validate_rows(invalid)

    def test_public_origin_stays_at_208_after_phase_two(self) -> None:
        self.assertEqual(self.data.origin_for_window(209), 208)
        self.assertEqual(self.data.origin_for_window(222), 208)
        self.assertEqual(self.data.complete_fold_origins()[-1], 195)
        self.data.historical_end = 221
        try:
            self.assertEqual(self.data.origin_for_window(209), 208)
            self.assertEqual(self.data.origin_for_window(222), 221)
            self.assertEqual(self.data.complete_fold_origins()[-1], 208)
        finally:
            self.data.historical_end = 208

    def test_state_decay_beats_historical_rate_on_thirteen_week_folds(self) -> None:
        summary = self.data.forecast_backtest()["summary"]
        self.assertLess(summary["brier_state_decay"], summary["brier_full"])
        self.assertLess(summary["brier_news_decay"], summary["brier_state_decay"])
        self.assertLess(summary["mape_price_blend"], summary["mape_last"])

    def test_news_candidate_changes_only_risk_and_ignores_future_headlines(self) -> None:
        state = self.data.candidate_rows(PROFILES["primary"])
        news = self.data.candidate_rows(PROFILES["news_primary"])
        self.assertTrue(any(a["disruption_prob"] != b["disruption_prob"] for a, b in zip(state, news)))
        for a, b in zip(state, news):
            for field in self.data.sample_header:
                if field != "disruption_prob":
                    self.assertEqual(a[field], b[field])
        before = self.data.warning_score("lithium", 208)
        self.data.news_by_week[209].append("Kestria lithium producers flag maintenance delays")
        self.data.warning_score.cache_clear()
        try:
            self.assertEqual(self.data.warning_score("lithium", 208), before)
        finally:
            self.data.news_by_week[209].pop()
            self.data.warning_score.cache_clear()


if __name__ == "__main__":
    unittest.main()
