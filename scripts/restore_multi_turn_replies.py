#!/usr/bin/env python3
"""Export intermediate reference replies from original Studio annotations.

Only conversation[i].ai_response for non-final turns is imported. Original
rendering prompts, service URLs, captions, metadata, and final answers are never
copied. The existing release supplies IDs, media paths, and final-turn rubrics.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def restore_rows(rows: list[dict], sources: dict[str, dict]) -> list[dict]:
    """Validate correspondence before returning any updated release records."""
    updated = copy.deepcopy(rows)
    seen = set()
    for row in updated:
        sample_id = row["id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate release ID: {sample_id}")
        seen.add(sample_id)
        source = sources.get(sample_id)
        if source is None or source.get("id") != sample_id:
            raise ValueError(f"Missing or mismatched source: {sample_id}")
        turns = row["turns"]
        conversation = source["conversation"]
        expected = list(range(1, row["num_turns"] + 1))
        if (
            len(turns) < 2
            or len(turns) != row["num_turns"]
            or len(conversation) != row["num_turns"]
            or source["scenario"]["total_rounds"] != row["num_turns"]
            or [turn["round"] for turn in turns] != expected
            or [turn["round"] for turn in conversation] != expected
        ):
            raise ValueError(f"Round count or ordering mismatch: {sample_id}")
        if conversation[-1].get("key_points") != row["key_points"]:
            raise ValueError(f"Final-turn rubric mismatch: {sample_id}")
        if turns[-1].get("reference_reply") is not None:
            raise ValueError(f"Final reference reply in release input: {sample_id}")
        for turn, original in zip(turns[:-1], conversation[:-1]):
            reply = original.get("ai_response")
            if not isinstance(reply, str) or not reply.strip():
                raise ValueError(
                    f"Missing intermediate reply: {sample_id}, round {turn['round']}"
                )
            turn["reference_reply"] = reply
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Existing release multi_turn.jsonl")
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Original Studio JSON directory, searched recursively")
    parser.add_argument("--output", type=Path, required=True,
                        help="Generated release JSONL; may be the same as --data")
    parser.add_argument("--examples-dir", type=Path, action="append", default=[],
                        help="Refresh existing multi-turn examples in this directory")
    args = parser.parse_args()

    sources = {}
    for path in sorted(args.source_dir.rglob("*.json")):
        source = json.loads(path.read_text(encoding="utf-8"))
        sample_id = source["id"]
        if sample_id in sources:
            raise ValueError(f"Duplicate source ID: {sample_id}")
        sources[sample_id] = source
    rows = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    updated = restore_rows(rows, sources)
    by_id = {row["id"]: row for row in updated}

    # Validate and prepare every example before writing generated artifacts.
    examples = []
    for directory in args.examples_dir:
        for path in sorted(directory.glob("*.json")):
            example = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(example, dict) or "num_turns" not in example:
                continue
            row = by_id[example["id"]]
            if any(example.get(key) != value for key, value in row.items()
                   if key != "turns"):
                raise ValueError(f"Example does not match its release row: {path.name}")
            old_turns = [
                {key: value for key, value in turn.items() if key != "reference_reply"}
                for turn in example["turns"]
            ]
            new_turns = [
                {key: value for key, value in turn.items() if key != "reference_reply"}
                for turn in row["turns"]
            ]
            if old_turns != new_turns:
                raise ValueError(f"Example media or rounds differ: {path.name}")
            example["turns"] = row["turns"]
            examples.append((path, json.dumps(example, ensure_ascii=False, indent=2) + "\n"))

    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in updated),
        encoding="utf-8",
    )
    for path, text in examples:
        path.write_text(text, encoding="utf-8")
    count = sum(len(row["turns"]) - 1 for row in updated)
    print(f"Exported {count} intermediate replies across {len(updated)} instances; "
          f"refreshed {len(examples)} examples.")


if __name__ == "__main__":
    main()
