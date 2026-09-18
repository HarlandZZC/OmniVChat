#!/usr/bin/env python3
"""Score model replies on OmniVChat-Bench.

This is the EVALUATION path. It reports five quantities per reply and aggregates them the way
the paper does. Only the first one is the benchmark number:

    rubric  r(y) in [0, 1]   THE REPORTED NUMBER. Fraction of rubric credit earned, under the
                             tier gate: tier 0 is a hard gate that scores no points but zeroes
                             the sample when missed, and tier t is only counted once every tier
                             below it is fully satisfied. Equation 1 of the paper. Aggregated as
                             the Subcategory Mean -- the unweighted mean over the 17
                             subcategory means, so a large subcategory does not dominate.

    format  f(y) in {0, 1}   Whether the completion is structurally valid: exactly one closing
                             '</think>' with a non-empty reply after it for a thinking
                             checkpoint, no reasoning tags at all for an instruct one. Set
                             REQUIRE_THINK_CLOSE=1 for a thinking checkpoint. Costs no judge
                             call. On a healthy run this sits at 1.000 and is only interesting
                             when it does not: a dip means truncated generations.

    style   s(y) in {0, 1}   Whether the reply obeys the style guidelines -- plain conversational
                             language, no markdown, no bullet points, no emoji, no padding. A
                             separate judge call against a fixed yardstick (reward/reward.py
                             holds the text). Speech-facing replies are the point of this
                             benchmark, and a model that answers correctly in a bulleted list has
                             not answered well. Costs ONE extra judge call per reply; --no-style
                             turns it off.

    words   integer          Length of the reply, counted so that Chinese and English land at a
                             comparable magnitude: one Han character is one word, one run of
                             Latin letters or digits is one word, punctuation is not counted.
                             Without that property a length term degenerates into a language
                             detector.

    eff     rubric/1k words  ABSOLUTE reply efficiency: rubric credit per thousand words. How
                             much of the rubric the reply earned per unit of length. Read it
                             together with `words`: two models at the same rubric score are not
                             equally good if one took three times as many words.

WHY eff IS NOT THE TRAINING EFFICIENCY TERM

Training uses a GROUP-RELATIVE efficiency: within one rollout group -- the N completions
sampled for the SAME prompt -- raw rubric/words is min-max normalised to [0, 1], so the term
asks "of these N answers to this one prompt, which earned its credit in the fewest words". That
number does not exist outside a rollout group, and it cannot be reconstructed from a file of
one reply per prompt. Cross-prompt comparison is deliberately avoided there: a question that
simply needs a longer answer would otherwise read as an inefficient one.

So this script reports the absolute quantity instead, which IS comparable across models on the
same benchmark, and says so rather than pretending to reproduce the training term. Same
underlying counter (count_words), different normalisation.

WHAT THIS SCRIPT DOES NOT DO

It does not combine the five into one number. The training reward does that --
`r(y) + 0.5*f(y) + 0*e(y) + 0*s(y)`, range [0, 1.5], Equation 4 -- and a combined figure is not
comparable with a benchmark score. See ../reward/ and the README.

The metric itself is not redefined here. Tier gating, the judge prompt, the rubric formatting,
the style yardstick and the word counter are all imported from ../reward/reward.py, which
is the single definition shared with the RL reward. Only the aggregation and the request loop
are local.

Usage
-----
    export JUDGE_API_BASE=https://your-endpoint.example/v1   # OpenAI-compatible
    export JUDGE_API_KEY=...
    export JUDGE_MODEL=<the judge model served there>

    # replies: JSONL with {"id": ..., "response": ...} per line, any order
    python3 score.py --data ../data/single_turn.jsonl --replies replies.jsonl
    python3 score.py --data ../data/multi_turn.jsonl  --replies replies.jsonl

    # rubric only, half the judge calls
    python3 score.py --data ../data/single_turn.jsonl --replies replies.jsonl --no-style

    # MSA-PLA only: score against the embodied-assistant rubric instead
    python3 score.py --data ../data/single_turn.jsonl --replies replies.jsonl \
                     --rubric key_points_embodied

Options that change the number
------------------------------
    --judge-model     the judge. Report which one produced a number: a weaker judge mostly
                      costs agreement on borderline criteria, but it is not the same metric.
    --target-model    name shown to the judge as the model under test. Matters for MSA-IC
                      subcategories, whose rubrics ask the model to state its own name. Set it,
                      and TARGET_IDENTITY_NAMES for the model's other acceptable names, or
                      those criteria score 0.
    REQUIRE_THINK_CLOSE=1
                      for a thinking checkpoint: the reply is what follows the last '</think>',
                      and a completion that never closed is a truncated trace, scored 0 rather
                      than handed to the judge as if it were an answer.

A reply whose rubric judge call never succeeds is reported as unjudged and excluded from the
mean, not counted as 0. The training path does the opposite -- it substitutes a running mean --
because there a 0 would punish a completion for an API failure.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reward"))
import reward as core  # noqa: E402  the one definition of the metric


def parse_hits(raw: str):
    """Pull the hit list out of the judge reply. None only if there is really no list.

    Two passes, because strict JSON is not reliable here: the judge quotes the model's own
    words inside "reasoning", and those quotes arrive unescaped often enough to matter. That
    breaks json.loads on a reply whose "hits" array is perfectly well formed, and dropping
    those samples would silently shrink the evaluation set.
    """
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j > i:
        try:
            hits = json.loads(s[i:j + 1]).get("hits")
            if isinstance(hits, list):
                return [int(h) for h in hits if isinstance(h, int) or str(h).isdigit()]
        except json.JSONDecodeError:
            pass                      # fall through; "reasoning" is not worth a parser

    m = re.search(r'"hits"\s*:\s*\[([^\]]*)\]', s)
    if m:
        return [int(x) for x in re.findall(r"\d+", m.group(1))]
    return None


def parse_compliant(raw: str):
    """True/False out of the style judge's {"compliant": ...}. None when it said neither.

    Regex rather than json.loads for the same reason as parse_hits: the value is what matters
    and a stray character elsewhere in the reply must not discard it.
    """
    m = re.search(r'"compliant"\s*:\s*(true|false)', raw, re.I)
    return m.group(1).lower() == "true" if m else None


class Judge:
    """One session, two questions: which rubric criteria were hit, and is the style compliant."""

    def __init__(self, model: str, url: str, key: str, concurrency: int, retries: int = 8):
        self.model, self.url, self.key, self.retries = model, url, key, retries
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=concurrency, pool_maxsize=concurrency, max_retries=0)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.calls = 0
        self._lock = threading.Lock()

    def _ask(self, system: str, user: str, parse, max_tokens: int):
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "max_tokens": max_tokens,
                   "temperature": core.JUDGE_TEMPERATURE}
        for attempt in range(1, self.retries + 1):
            try:
                r = self.session.post(
                    self.url, json=payload,
                    headers={"Authorization": f"Bearer {self.key}"}, timeout=120)
            except requests.RequestException:
                r = None
            if r is not None and r.status_code == 200:
                with self._lock:
                    self.calls += 1
                got = parse(r.json()["choices"][0]["message"]["content"] or "")
                if got is not None:
                    return got
            elif r is not None and r.status_code in (400, 401, 403, 404):
                return None          # not transient; retrying only burns quota
            if attempt < self.retries:
                time.sleep(min(30.0, 1.5 * 2 ** (attempt - 1)) * (0.7 + 0.6 * random.random()))
        return None

    def hits_for(self, reply: str, key_points: list):
        return self._ask(core.get_judge_system_prompt(),
                         core.build_judge_user_message(reply, key_points),
                         parse_hits, core.JUDGE_MAX_TOKENS)

    def style_ok(self, reply: str):
        # 64 tokens is plenty for {"compliant": true}; capping it keeps the extra call cheap.
        return self._ask(core.SYSPROMPT_JUDGE_SYSTEM,
                         core.build_sysprompt_judge_message(reply),
                         parse_compliant, 64)


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data/single_turn.jsonl or data/multi_turn.jsonl")
    ap.add_argument("--replies", required=True, help='JSONL of {"id","response"}')
    ap.add_argument("--rubric", default="key_points",
                    help="rubric field to score against (key_points | key_points_embodied)")
    ap.add_argument("--judge-model", default=core.JUDGE_MODEL)
    ap.add_argument("--judge-url", default=core.JUDGE_API_URL)
    ap.add_argument("--target-model", default=core.TARGET_MODEL_NAME)
    ap.add_argument("--no-style", action="store_true",
                    help="skip the style judge call (halves the number of requests)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", type=Path, help="write per-sample scores here")
    args = ap.parse_args()

    key = os.environ.get("JUDGE_API_KEY", "")
    missing = [n for n, v in (("JUDGE_API_KEY", key), ("JUDGE_API_BASE", args.judge_url),
                              ("JUDGE_MODEL", args.judge_model)) if not v]
    if missing:
        sys.exit("the judge is not configured -- %s unset. See the module docstring."
                 % ", ".join(missing))
    core.TARGET_MODEL_NAME = args.target_model
    want_style = not args.no_style

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    replies = {}
    for l in Path(args.replies).read_text(encoding="utf-8").splitlines():
        if l.strip():
            o = json.loads(l)
            replies[o["id"]] = o.get("response") or ""

    todo = []
    for r in rows:
        kps = r.get(args.rubric)
        if not kps:
            continue                      # e.g. key_points_embodied exists for MSA-PLA only
        todo.append((r, kps, replies.get(r["id"])))
    missing_replies = [r["id"] for r, _, rep in todo if rep is None]
    todo = [(r, k, rep) for r, k, rep in todo if rep is not None]
    print(f"{len(rows)} samples in {Path(args.data).name}; scoring {len(todo)} "
          f"against '{args.rubric}'"
          + (f"; {len(missing_replies)} without a reply (skipped)" if missing_replies else "")
          + (f"\njudge {args.judge_model}, {'rubric + style' if want_style else 'rubric only'}"
             f" -> up to {len(todo) * (2 if want_style else 1)} calls"
             f"; think-close={'on' if core.REQUIRE_THINK_CLOSE else 'off'}"))
    if not todo:
        sys.exit("nothing to score")

    judge = Judge(args.judge_model, args.judge_url, key, args.concurrency)
    results, unjudged, style_unjudged = [], [], []
    lock, done = threading.Lock(), [0]

    def work(item):
        row, kps, raw_reply = item
        # format grades the RAW completion (that is where the tags are); everything else grades
        # the reply, which for a thinking checkpoint is what follows the last '</think>'.
        fmt = core.format_score(raw_reply)
        reply = core.strip_thinking(raw_reply)
        words = core.count_words(reply)
        # An empty reply is a real 0, not a judge failure: no need to spend a call on either question.
        if not reply.strip():
            hits, style = [], (False if want_style else None)
        else:
            hits = judge.hits_for(reply, kps)
            style = judge.style_ok(reply) if want_style else None
        with lock:
            if hits is None:
                unjudged.append(row["id"])
            else:
                sc = core.compute_tier_gated_score(kps, hits)
                rubric = core.hit_rate(sc)
                if want_style and style is None:
                    style_unjudged.append(row["id"])
                results.append({
                    "id": row["id"], "subcategory": row["subcategory"], "ability": row["ability"],
                    "rubric": rubric, "format": fmt,
                    "style": None if style is None else float(style),
                    "words": words,
                    # Absolute efficiency, per 1,000 words so the figure is readable. NOT the
                    # group-relative training term -- see the module docstring.
                    "eff_per_1k": 1000.0 * rubric / max(words, 1),
                    "gate_passed": sc["gate_passed"], "hits": hits,
                })
            done[0] += 1
            if done[0] % 100 == 0 or done[0] == len(todo):
                print(f"  {done[0]}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for fut in as_completed([pool.submit(work, t) for t in todo]):
            fut.result()

    if not results:
        sys.exit("every judge call failed")

    def agg(rs: list) -> dict:
        styles = [x["style"] for x in rs if x["style"] is not None]
        return {"n": len(rs), "rubric": mean(x["rubric"] for x in rs),
                "format": mean(x["format"] for x in rs),
                "style": mean(styles) if styles else None,
                "words": mean(x["words"] for x in rs),
                "eff": mean(x["eff_per_1k"] for x in rs)}

    by_sub, by_ab = defaultdict(list), defaultdict(list)
    for x in results:
        by_sub[x["subcategory"]].append(x)
        by_ab[x["ability"]].append(x)
    sub_agg = {k: agg(v) for k, v in by_sub.items()}
    ab_agg = {k: agg(v) for k, v in by_ab.items()}
    pooled = agg(results)
    # The reported Mean is the unweighted mean over subcategory means, not over samples.
    subcat_mean = {k: mean(sub_agg[s][k] for s in sub_agg)
                   for k in ("rubric", "format", "words", "eff")}
    styled = [sub_agg[s]["style"] for s in sub_agg if sub_agg[s]["style"] is not None]
    subcat_mean["style"] = mean(styled) if styled else None

    def fmt_row(name: str, a: dict, n=None) -> str:
        st = "     -" if a["style"] is None else f"{a['style']:>6.3f}"
        return (f"{name:<18}{(n if n is not None else a['n']):>5}{a['rubric']:>9.4f}"
                f"{a['format']:>8.3f}{st}{a['words']:>8.1f}{a['eff']:>8.2f}")

    head = (f"\n{'':<18}{'n':>5}{'rubric':>9}{'format':>8}{'style':>6}{'words':>8}{'eff':>8}")
    print(head)
    print(f"{'':<18}{'':>5}{'[0,1]':>9}{'{0,1}':>8}{'{0,1}':>6}{'count':>8}{'/1k':>8}")
    print("-" * 62)
    for k in sorted(sub_agg):
        print(fmt_row(k, sub_agg[k]))
    print("-" * 62)
    for k in sorted(ab_agg):
        print(fmt_row(k, ab_agg[k]))
    print("-" * 62)
    print(fmt_row("Pooled Mean", pooled))
    print(fmt_row("Mean (subcat)", subcat_mean, n=len(sub_agg))
          + "   <- rubric here is the reported number")
    print("\nrubric = Equation 1, tier-gated, [0,1] -- the benchmark score."
          "\nformat/style are {0,1} per reply, averaged. eff = rubric credit per 1,000 words,"
          "\nabsolute (NOT the group-relative training term). None of these is added into rubric.")
    if unjudged:
        print(f"\nunjudged and excluded from the mean: {len(unjudged)}")
    if style_unjudged:
        print(f"style unjudged (counted in rubric, absent from the style mean): {len(style_unjudged)}")
    print(f"judge calls: {judge.calls}")

    if args.out:
        args.out.write_text(json.dumps(
            {"judge_model": args.judge_model, "rubric_field": args.rubric,
             "judge_prompt_sha": core.JUDGE_PROMPT_SHA,
             "reported_mean": subcat_mean["rubric"],
             "mean_subcategory": subcat_mean, "pooled_mean": pooled,
             "n_scored": len(results), "n_unjudged": len(unjudged),
             "n_style_unjudged": len(style_unjudged),
             "subcategory": sub_agg, "ability": ab_agg,
             "per_sample": results}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
