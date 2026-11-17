"""Check fixed history, final-answer exclusion, and safe annotation export."""
import copy
import unittest

from eval.multi_turn import build_messages
from scripts.restore_multi_turn_replies import restore_rows


def sample():
    return {
        "id": "example",
        "ability": "MEA",
        "subcategory": "MEA-RC-MRR",
        "num_turns": 3,
        "turns": [
            {"round": 1, "video": "media/1.mp4", "reference_reply": "First reply."},
            {"round": 2, "video": "media/2.mp4", "reference_reply": "Second reply."},
            {"round": 3, "video": "media/3.mp4"},
        ],
        "key_points": [{"tier": 0, "point": "Reply in English."}],
    }


def original(row):
    return {
        "id": row["id"],
        "scenario": {"total_rounds": row["num_turns"]},
        "conversation": [
            {"round": 1, "ai_response": "First reply.", "video": {"private": "metadata"}},
            {"round": 2, "ai_response": "Second reply."},
            {"round": 3, "ai_response": "Final answer must stay out of the prompt.",
             "key_points": row["key_points"]},
        ],
    }


class ContextTests(unittest.TestCase):
    def test_order_adapter_and_no_rubric_leakage(self):
        row = sample()
        before = copy.deepcopy(row)
        messages = build_messages(row, lambda path: [{"encoded_video": path}])
        self.assertEqual(messages, [
            {"role": "user", "content": [{"encoded_video": "media/1.mp4"}]},
            {"role": "assistant", "content": "First reply."},
            {"role": "user", "content": [{"encoded_video": "media/2.mp4"}]},
            {"role": "assistant", "content": "Second reply."},
            {"role": "user", "content": [{"encoded_video": "media/3.mp4"}]},
        ])
        self.assertEqual(row, before)

    def test_invalid_history_rejected_before_adapter_runs(self):
        for missing in (None, "", "  ", 1):
            with self.subTest(reference_reply=missing):
                row = sample()
                row["turns"][1]["reference_reply"] = missing
                calls = []
                with self.assertRaises(ValueError):
                    build_messages(row, calls.append)
                self.assertEqual(calls, [])

    def test_final_reference_rejected_but_loader_null_allowed(self):
        row = sample()
        row["turns"][-1]["reference_reply"] = "Leaked final answer."
        with self.assertRaises(ValueError):
            build_messages(row, str)
        row["turns"][-1]["reference_reply"] = None
        self.assertEqual(build_messages(row, str)[-1]["role"], "user")

    def test_bad_rounds_and_missing_media_rejected(self):
        for change in ("count", "order", "video"):
            with self.subTest(change=change):
                row = sample()
                if change == "count":
                    row["num_turns"] = 2
                elif change == "order":
                    row["turns"][1]["round"] = 1
                else:
                    row["turns"][2].pop("video")
                with self.assertRaises(ValueError):
                    build_messages(row, str)


class ExportTests(unittest.TestCase):
    def test_only_intermediate_replies_imported_and_idempotent(self):
        row = sample()
        for turn in row["turns"]:
            turn.pop("reference_reply", None)
        before = copy.deepcopy(row)
        source = original(row)
        updated = restore_rows([row], {"example": source})
        self.assertEqual(updated, [sample()])
        self.assertEqual(row, before)
        self.assertEqual(restore_rows(updated, {"example": source}), updated)

    def test_mismatched_sources_fail(self):
        for change in ("missing_id", "round_count", "order", "rubric", "empty_reply"):
            with self.subTest(change=change):
                row = sample()
                source = original(row)
                if change == "round_count":
                    source["scenario"]["total_rounds"] = 4
                elif change == "order":
                    source["conversation"].reverse()
                elif change == "rubric":
                    source["conversation"][-1]["key_points"] = []
                elif change == "empty_reply":
                    source["conversation"][0]["ai_response"] = ""
                sources = {} if change == "missing_id" else {"example": source}
                with self.assertRaises(ValueError):
                    restore_rows([row], sources)


if __name__ == "__main__":
    unittest.main()
