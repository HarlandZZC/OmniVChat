"""Build fixed reference history for final-turn OmniVChat evaluation.

The caller supplies an adapter that encodes a video's path, including its audio
track, in the target model's native user-content format. Only media paths and
intermediate reference replies enter the messages; rubrics remain with the scorer.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_messages(
    row: dict[str, Any],
    make_user_content: Callable[[str], Any],
) -> list[dict[str, Any]]:
    """Return alternating user/assistant messages ending with the final user turn.

    ``make_user_content`` receives each repository-relative video path. It must
    preserve the video's audio. Resolve paths against the downloaded dataset root.
    Missing history is an error: generating substitute replies changes the benchmark.
    """
    turns = row.get("turns")
    if not isinstance(turns, list) or len(turns) < 2:
        raise ValueError("A multi-turn instance must contain at least two turns")
    if type(row.get("num_turns")) is not int or row["num_turns"] != len(turns):
        raise ValueError("num_turns must equal the number of turns")

    # Validate the whole record before invoking the model-specific media adapter.
    for index, turn in enumerate(turns, 1):
        if not isinstance(turn, dict) or type(turn.get("round")) is not int:
            raise ValueError(f"Turn {index} must have an integer round")
        if turn["round"] != index:
            raise ValueError("Rounds must be ordered consecutively starting at 1")
        if not isinstance(turn.get("video"), str) or not turn["video"].strip():
            raise ValueError(f"Turn {index} is missing its video path")
        reference = turn.get("reference_reply")
        if index < len(turns):
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(f"Turn {index} is missing its reference_reply")
        elif reference is not None:
            # Some dataset loaders materialize an absent struct field as null.
            raise ValueError("The final turn must not contain a reference_reply")

    messages = []
    for index, turn in enumerate(turns):
        messages.append({
            "role": "user",
            "content": make_user_content(turn["video"]),
        })
        if index < len(turns) - 1:
            messages.append({
                "role": "assistant",
                "content": turn["reference_reply"],
            })
    return messages
