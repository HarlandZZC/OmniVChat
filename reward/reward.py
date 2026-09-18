"""Judging and tier-gated scoring for OmniVChat-RL.

This module is the definition of the metric, and the only file that defines it. Evaluation
(eval/score.py) and GSPO training (through the optional trainer_adapter.py) both import it, so the
two cannot drift apart.

Self-contained on purpose: a training run must not change its reward because some other file
was edited mid-flight. Everything the judge sees lives in this directory, including the two
prompt files under prompts/.

WHAT TRAINING ADDS ON TOP OF THE RUBRIC

    combine_score = hit_rate + FORMAT_WEIGHT * format_score
                             + WORD_EFF_WEIGHT * word_eff
                             + SYSPROMPT_WEIGHT * sysprompt

Template defaults are 1 : 0.5 : 0 : 0, so the training reward runs to 1.5. Evaluation
deliberately reports hit_rate alone, range [0, 1]: that is the benchmark number, and it stays
comparable across reward designs precisely because it ignores these extra terms. Setting
FORMAT_WEIGHT=0 makes combine_score identical to hit_rate again.

word_eff cannot be computed in this file. It is relative WITHIN a rollout group -- it needs
the ROLLOUT_N completions of the same prompt to compare against -- so this file holds only
the definition (count_words) and the weight, and the actual normalisation happens in
trainer_adapter.py.

JUDGE-FAILURE POLICY DIFFERS BETWEEN THE TWO PATHS, ON PURPOSE

Evaluation excludes a sample whose judge call never succeeded from the mean. Training
substitutes a running mean of the samples already scored in this process. Scoring an API
failure as 0 during training would punish a completion for something it did not do, and a
handful of those is enough to distort the advantage for a whole step.

REQUIRE_THINK_CLOSE SELECTS HOW COMPLETIONS ARE PARSED

Set it to 0 for an instruct checkpoint and 1 for a thinking checkpoint. A thinking chat
template appends '<think>\n' to the prompt, so the opening tag never appears in the
completion; a missing closing tag then means the reasoning was cut off by the token budget
and no reply was produced, which must be recorded as an empty string (score 0). Getting this
wrong feeds a bare reasoning trace to the judge as if it were a reply. See strip_thinking.
"""
import asyncio
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

import aiohttp
import jinja2

# Name of the model under test. It goes into the judge's [Model Under Test] field and is what
# a <model_name> criterion in key_points is judged against. The default is a placeholder: set
# it to the model you are testing, or every such criterion scores 0. This matters for the
# MSA-IC subcategories, whose rubrics ask the model to say who it is.
TARGET_MODEL_NAME = os.environ.get("TARGET_MODEL_NAME", "omni-model")

# Other names the same model may correctly use for itself -- a short product name, a familiar
# abbreviation, the name in another language. Comma-separated. Leave the primary name out: it
# is added automatically, and a duplicate here is dropped by case-insensitive de-duplication.
TARGET_IDENTITY_NAMES = [
    n.strip() for n in os.environ.get("TARGET_IDENTITY_NAMES", "").split(",")
    if n.strip()
]

# System prompt given to the model under test. The data builders use it when generating
# training and evaluation data, so training and inference always run under the same prompt --
# the system message inside the data is its only copy.
#
# This value is per-variant in the training setup and normally comes from the environment.
# The default below is only the fallback for when nothing exported it. A run that silently
# falls back here would produce data that looks completely normal but carries the wrong
# prompt, which is why the training-side loaders fail loudly on a missing value instead.
TARGET_SYSTEM_PROMPT = os.environ.get(
    "TARGET_SYSTEM_PROMPT",
    "You are a helpful AI assistant having a conversation with the user. "
    "Watch and listen to the video carefully, and then respond naturally.",
)

# The judge is any OpenAI-compatible chat-completions service. Nothing below is tied to a
# particular provider or a particular model:
#
#     export JUDGE_API_BASE=https://your-endpoint.example/v1
#     export JUDGE_API_KEY=...
#     export JUDGE_MODEL=<the judge model served there>
#
# JUDGE_API_URL overrides the assembled URL for a service that does not use the /v1 layout.
# The published numbers came from a large instruction-tuned judge at temperature 0.1. A weaker
# judge mostly costs agreement on the borderline criteria rather than shifting the mean, but it
# is not the same metric, so report which judge produced a number.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "")
JUDGE_API_BASE = os.environ.get("JUDGE_API_BASE", "").rstrip("/")
JUDGE_API_URL = os.environ.get("JUDGE_API_URL") or (
    f"{JUDGE_API_BASE}/chat/completions" if JUDGE_API_BASE else "")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "")
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "2048"))
JUDGE_TEMPERATURE = float(os.environ.get("JUDGE_TEMPERATURE", "0.1"))
MAX_RETRIES = int(os.environ.get("JUDGE_MAX_RETRIES", "20"))
# Hosted endpoints answer concurrency bursts with HTTP 429 and ask clients to spread requests
# out, hence the cap. In training each reward worker process holds its own, so the effective
# ceiling is JUDGE_MAX_CONCURRENCY x number of worker processes.
JUDGE_MAX_CONCURRENCY = int(os.environ.get("JUDGE_MAX_CONCURRENCY", "4"))
JUDGE_START_JITTER = float(os.environ.get("JUDGE_START_JITTER", "0.5"))


def require_judge_config() -> None:
    """Fail immediately when the judge is not fully configured.

    All three of key, URL and model are needed. Otherwise every reward returns 0 without
    complaint, the advantage is flat for the whole run, and nothing in the logs says why.
    """
    missing = [name for name, value in (
        ("JUDGE_API_KEY", JUDGE_API_KEY),
        ("JUDGE_API_BASE (or JUDGE_API_URL)", JUDGE_API_URL),
        ("JUDGE_MODEL", JUDGE_MODEL),
    ) if not value]
    if missing:
        raise RuntimeError(
            "the judge is not configured -- %s unset. The judge cannot be called and every "
            "reward would silently be 0. Export them before running." % ", ".join(missing)
        )

PROMPTS_DIR = Path(__file__).parent / "prompts"


def get_judge_system_prompt() -> str:
    return (PROMPTS_DIR / "judge_system_prompt.md").read_text(encoding="utf-8")


def render(template_name: str, context: dict) -> str:
    """Render prompts/<template_name>.md.j2."""
    path = PROMPTS_DIR / f"{template_name}.md.j2"
    if not path.exists():
        raise FileNotFoundError(f"template not found: {path}")
    env = jinja2.Environment(undefined=jinja2.Undefined)
    return env.from_string(path.read_text(encoding="utf-8")).render(**context)


JUDGE_SYSTEM_PROMPT = get_judge_system_prompt()
# The prompts are read once per process at import, and a crashed training run auto-resumes. If
# the judge prompt was edited between two launches, the first and second halves of one
# experiment were scored by different rewards. Recording a fingerprint makes that findable
# afterwards. It covers the system prompt and the user template, so either edit changes it.
_judge_prompt_material = JUDGE_SYSTEM_PROMPT + (
    PROMPTS_DIR / "judge_user_prompt.md.j2").read_text(encoding="utf-8")
JUDGE_PROMPT_SHA = hashlib.sha256(_judge_prompt_material.encode("utf-8")).hexdigest()[:12]


# Where the '<think>' opening tag lives depends on the checkpoint, and that is the one thing a
# tag-based rule cannot infer from the text alone — hence a flag rather than two forked copies of
# this module. Set it per checkpoint; see strip_thinking() for what each value means.
REQUIRE_THINK_CLOSE = os.environ.get("REQUIRE_THINK_CLOSE", "0") == "1"


def strip_thinking(text: str) -> str:
    """Extract the reply that follows the reasoning block, if there is one.

    The reply is always whatever follows the last '</think>'. What a *missing* '</think>' means is
    what differs between checkpoints, and it is not inferable from the text:

    REQUIRE_THINK_CLOSE=0  (an instruct checkpoint, or any pipeline whose decode step re-prepends
                            the response prefix so the opener lands in the completion)
        The opening tag, if the model reasons at all, is part of the completion. So no tags at all
        means the model simply answered, and the text IS the reply. A truncated trace is still
        caught, because it carries the opener.

    REQUIRE_THINK_CLOSE=1  (think)
        The chat template appends '<think>\\n' to the generation prompt, so the opener is in the
        PROMPT and never in the completion. A completion with no '</think>' is therefore a reasoning
        block that ran out of MAX_RESPONSE_LENGTH before producing any reply. Returning text.strip()
        there would hand the raw reasoning trace to the judge as if it were the reply, so it must be
        '' (scored 0). The training adapter surfaces this as reply_ok, keeping the truncation rate
        visible in the logs rather than hidden inside a rubric score of 0.
    """
    if not text:
        return ""
    idx = text.rfind("</think>")
    if idx != -1:
        return text[idx + len("</think>"):].strip()
    if REQUIRE_THINK_CLOSE or "<think>" in text:
        return ""
    return text.strip()


# Weight of the format term. It affects the TRAINING reward only, never evaluation.
# Set it to 0 and combine_score becomes identical to hit_rate, i.e. exactly the reward from
# before the format term existed. To run a with/without-format ablation, change this one
# number rather than editing the adapter.
FORMAT_WEIGHT = float(os.environ.get("FORMAT_WEIGHT", "0.5"))


def format_score(text: str) -> float:
    """Whether the completion is structurally valid: 1.0 or 0.0, binary, no partial credit.

    This takes the RAW completion, not the output of strip_thinking. What it judges is how the
    tags themselves are placed, and stripping them would leave nothing to judge. As in
    strip_thinking, the one thing that cannot be inferred from the text alone is where the
    opening tag lives, so it branches on REQUIRE_THINK_CLOSE too:

    REQUIRE_THINK_CLOSE=1  (think)
        The chat template appends '<think>\\n' to the prompt, so a valid completion is
        `reasoning + '</think>' + non-empty reply`: no opening tag should appear again (one that
        does means the model started a second reasoning block), there must be exactly one
        closing tag (more than one means nested or repeated blocks), and there must be a
        non-empty reply after it.

    REQUIRE_THINK_CLOSE=0  (instruct)
        This checkpoint does not produce reasoning blocks at all, so the completion should be
        the reply and nothing else. Any '<think>' or '</think>' is off-track, and a non-empty
        reply is the only requirement.

    Relation to reply_ok: the two overlap heavily on real data but are not the same question.
    reply_ok asks whether anything survives strip_thinking; this asks whether the tags are
    placed correctly. The difference is the cases strip_thinking accepts and this scores 0 --
    several '</think>', or a second '<think>' inside a thinking-variant completion --
    because strip_thinking takes the text after the LAST closing tag and would pass those to
    the judge as an ordinary reply.

    Measured on 1056 real completions from a thinking run: 1047 passed, 9 failed, a pass rate
    of 0.9915, and all 9 failures were truncated reasoning missing '</think>'. Zero cases of a
    stray opening tag, multiple closing tags, or an empty reply after a valid close. So on that
    model this term is very nearly reply_ok, and since the advantage is normalised within a
    rollout group, a +0.5 that every member earns is a constant that cancels out -- it only
    produces gradient when one group contains both truncated and untruncated samples. The
    reason to include it is to make "structurally off-track" an explicit term in the reward
    rather than something strip_thinking happens to absorb, not to lift scores.
    """
    if not text:
        return 0.0
    n_open = text.count("<think>")
    n_close = text.count("</think>")
    if REQUIRE_THINK_CLOSE:
        if n_open != 0 or n_close != 1:
            return 0.0
        idx = text.rfind("</think>")
        return 1.0 if text[idx + len("</think>"):].strip() else 0.0
    if n_open or n_close:
        return 0.0
    return 1.0 if text.strip() else 0.0


# Weight of the word-efficiency term. word_eff is already normalised within the rollout group
# to [0,1], so this is the most it can contribute. Set it to 0 to switch the term off, which
# also skips the grouping barrier it needs.
WORD_EFF_WEIGHT = float(os.environ.get("WORD_EFF_WEIGHT", "0"))

# ---------------------------------------------------------------------------
# sysprompt: ask the judge whether the reply obeys the style constraints. 1 if it does, 0 if not.
#
# The text below IS the grading standard, deliberately hard-coded here rather than read from
# TARGET_SYSTEM_PROMPT. The two are separate things: TARGET_SYSTEM_PROMPT is the system message
# fed to the model being trained, while this is the yardstick used to score it. Tying them
# together would have two bad effects -- editing the data prompt would silently change what the
# reward means, and measuring style adherence under a neutral prompt would become impossible.
# So even for a variant whose data carries no such system prompt, this term scores against the
# same yardstick.
#
# Default weight 0, so the template behaves exactly as it did before this term existed. Set
# SYSPROMPT_WEIGHT to enable it; note it costs one extra judge call per sample.
SYSPROMPT_WEIGHT = float(os.environ.get("SYSPROMPT_WEIGHT", "0"))

STYLE_GUIDELINES = (
    'Please strictly follow the following guidelines when generating responses. Avoid using any form'
    'atting markers, special symbols, or structured layouts. Do not include bold, italic, numbering,'
    ' bullet points, emojis, or other visual elements. The response must be natural conversational l'
    'anguage with smooth sentences and a human-like dialogue flow. Use standard punctuation—such as '
    'periods, commas, and question marks—to separate ideas clearly. Refrain from complex sentence st'
    'ructures, and above all, avoid redundant or wordy expressions. Be concise and direct. When list'
    'ing information, use continuous narration instead of bullet points.'
)

SYSPROMPT_JUDGE_SYSTEM = (
    "You are a strict style compliance checker. You are given a set of writing guidelines and one "
    "response. Decide whether the response OBEYS EVERY guideline. Judge only the style, never the "
    "factual content, and never the length of the underlying task. Reply with JSON only, exactly "
    '{"compliant": true} or {"compliant": false}, and nothing else.'
)


def build_sysprompt_judge_message(reply: str) -> str:
    """The standard plus the reply, handed to the same judge model."""
    return ("Guidelines:\n%s\n\nResponse to check:\n%s\n\n"
            "Does the response obey EVERY guideline above? Reply with JSON only: "
            '{"compliant": true} or {"compliant": false}.') % (STYLE_GUIDELINES, reply)


# CJK unified ideographs, including extensions A/B and the compatibility block. Chinese has no
# spaces, so it is counted per character; Latin text is split on whitespace.
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0002a6df]")
# Apostrophes and hyphens do not split a word: "don't" and "state-of-the-art" each count as one.
_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*")


def count_words(text: str) -> int:
    """Word count that holds for mixed Chinese and English.

    This dataset is roughly half and half, so splitting on whitespace alone will not do. One
    Han character counts as one word; one run of Latin letters or digits counts as one word.
    This is not linguistic tokenisation. It only has to satisfy one property: the same reply
    written in Chinese and in English must come out at a comparable magnitude. Otherwise
    word_eff degenerates into a language detector that rewards answering in Chinese as if that
    were efficiency. Punctuation is not counted.
    """
    if not text:
        return 0
    return len(_CJK_RE.findall(text)) + len(_LATIN_WORD_RE.findall(text))


def combine_score(rubric: float, fmt: float, word_eff: float = 0.0,
                  sysprompt: float = 0.0) -> float:
    """The final TRAINING reward. Ratio 1 : FORMAT : WORD_EFF : SYSPROMPT.

    Template defaults are 1 : 0.5 : 0 : 0, range [0, 1.5]. word_eff and sysprompt each have to
    be switched on deliberately because each has a cost: the first forces single-worker reward
    computation, the second adds one judge call per sample.

    Only the training adapter calls this. Evaluation deliberately does not -- see the top of
    this file.
    """
    return (rubric + FORMAT_WEIGHT * fmt + WORD_EFF_WEIGHT * word_eff
            + SYSPROMPT_WEIGHT * sysprompt)


def parse_key_points(solution) -> list:
    """Get key_points out of a solution, tolerating a JSON string, a dict, or malformed data.

    Training receives the trainer's ground_truth field (a JSON string) and evaluation receives
    the solution field of the eval data. Different shapes, same meaning, so the parsing lives in
    the shared module. On failure this returns an empty list and lets the caller score 0; it
    never raises, because on the training side one exception propagates through asyncio.gather
    and kills the whole step.
    """
    if solution is None:
        return []
    try:
        data = json.loads(solution) if isinstance(solution, str) else solution
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    key_points = data.get("key_points") or []
    return key_points if isinstance(key_points, list) else []


def key_point_tier(kp) -> int:
    if isinstance(kp, dict):
        try:
            return int(kp.get("tier", 1))
        except (TypeError, ValueError):
            return 1
    return 1


def format_key_points(key_points: list) -> str:
    """Format key_points as a numbered list."""
    lines = []
    for i, kp in enumerate(key_points, 1):
        if isinstance(kp, dict):
            point = kp.get("point", str(kp))
            tier = kp.get("tier", "")
            # Note that tier == 0 is falsy, so a gating criterion deliberately carries no
            # "[tier 0]" label. This is what the judge input is defined to look like, and
            # results are only comparable while it stays that way. Do not "tidy" this into
            # an `is not None` check.
            tier_str = f" [tier {tier}]" if tier else ""
            lines.append(f"{i}.{tier_str} {point}")
        else:
            lines.append(f"{i}. {kp}")
    return "\n".join(lines)


def identity_aliases() -> list:
    """Other correct names for the same model, minus any duplicate of the primary name.

    Case-insensitive de-duplication. Assembling the parentheses is the judge_user_prompt
    template's job; doing it here as well would let the two drift apart.
    """
    aliases = []
    seen = {TARGET_MODEL_NAME.strip().lower()}
    for name in TARGET_IDENTITY_NAMES:
        key = name.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        aliases.append(name.strip())
    return aliases


def build_judge_user_message(reply: str, key_points: list) -> str:
    """Render through this directory's judge_user_prompt template."""
    return render("judge_user_prompt", {
        "target_response": reply,
        "target_model": TARGET_MODEL_NAME,
        "identity_aliases": identity_aliases(),
        "key_points": format_key_points(key_points),
    })


def compute_tier_gated_score(key_points: list, hits: list) -> dict:
    """Tier-gated scoring.

    A tier contributes only when every earlier tier is fully met. Tier 0 earns no points but
    gates everything above it.
    """
    hit_indices = {int(h) for h in hits if isinstance(h, int) or str(h).isdigit()}
    tiers: dict = {}
    for idx, kp in enumerate(key_points, 1):
        tiers.setdefault(key_point_tier(kp), []).append(idx)

    # Tier 0 is a hard gate: any tier-0 criterion missed means 0 for the whole sample, with
    # tier 1 and above all locked. Tier 0 is only a gate -- meeting it adds no points and it
    # does not count toward the denominator, which covers tier >= 1 only.
    gate_indices = tiers.get(0, [])
    gate_passed = all(i in hit_indices for i in gate_indices)

    max_unlocked_tier = 0
    tier_summary = {}
    if gate_indices:
        tier_summary["0"] = {
            "total": len(gate_indices),
            "hits": sorted(i for i in gate_indices if i in hit_indices),
            "all_hit": gate_passed,
            "prerequisite_met": True,
            "counts": False,
        }

    for tier in sorted(t for t in tiers if t >= 1):
        indices = tiers[tier]
        hit_in_tier = sorted(i for i in indices if i in hit_indices)
        all_hit = all(i in hit_indices for i in indices)
        # If the gate failed, no scoring tier unlocks. Once it passes, tiers unlock one at a
        # time starting from tier 1.
        prerequisite_met = gate_passed and (tier == 1 or max_unlocked_tier == tier - 1)
        counts = prerequisite_met
        if prerequisite_met and all_hit:
            max_unlocked_tier = tier
        tier_summary[str(tier)] = {
            "total": len(indices),
            "hits": hit_in_tier,
            "all_hit": all_hit,
            "prerequisite_met": prerequisite_met,
            "counts": counts,
        }

    counted_hits = []
    gated_out_hits = []
    for idx in sorted(hit_indices):
        tier = key_point_tier(key_points[idx - 1]) if 1 <= idx <= len(key_points) else 1
        # A tier-0 hit never scores; a tier >= 1 hit scores only once that tier has unlocked.
        if tier >= 1 and tier_summary.get(str(tier), {}).get("counts", False):
            counted_hits.append(idx)
        else:
            gated_out_hits.append(idx)

    scorable_keypoints = sum(len(tiers[t]) for t in tiers if t >= 1)

    return {
        "score": len(counted_hits),
        "counted_hits": counted_hits,
        "gated_out_hits": gated_out_hits,
        "max_unlocked_tier": max_unlocked_tier,
        "tier_summary": tier_summary,
        "gate_passed": gate_passed,
        "scorable_keypoints": scorable_keypoints,
    }


def hit_rate(scoring: dict) -> float:
    """Normalise to [0, 1]: score divided by the number of scorable criteria."""
    total = scoring["scorable_keypoints"]
    return scoring["score"] / total if total else 0.0


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter.

    Without the random factor every call throttled by the same burst retries at the same instant and
    re-triggers the throttle, which is what the endpoint's Throttling.BurstRate message asks clients
    to avoid.
    """
    return 2 ** min(attempt, 5) * random.uniform(0.5, 1.5)


async def call_sysprompt_judge(session: aiohttp.ClientSession, reply: str) -> Optional[float]:
    """1.0 or 0.0: whether the reply obeys STYLE_GUIDELINES. None when retries run out.

    Retry budget, backoff, and giving up immediately on 400/401/403/404 all match call_judge --
    same endpoint, same throttling. Returning None lets the caller treat this term as "not
    judged" rather than passing it off as 0. A 0 means "violated the style"; None means "never
    got an answer". Conflating them makes the reward drop for no reason whenever the judge is
    flaky.
    """
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": SYSPROMPT_JUDGE_SYSTEM},
            {"role": "user", "content": build_sysprompt_judge_message(reply)},
        ],
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": 64,
        "enable_thinking": False,
    }
    headers = {"Authorization": f"Bearer {JUDGE_API_KEY}", "Content-Type": "application/json"}
    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(
                JUDGE_API_URL, headers=headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:200]
                    if resp.status in (400, 401, 403, 404):
                        print(f"  [sysprompt] HTTP {resp.status} is not retryable: {body}",
                              file=sys.stderr)
                        return None
                    print(f"  [sysprompt] HTTP {resp.status} "
                          f"(attempt {attempt+1}/{MAX_RETRIES}): {body}", file=sys.stderr)
                    await asyncio.sleep(_backoff(attempt))
                    continue
                content = (await resp.json())["choices"][0]["message"]["content"]
                m = re.search(r"\{.*\}", content, re.DOTALL)
                if m:
                    v = json.loads(m.group()).get("compliant")
                    if isinstance(v, bool):
                        return 1.0 if v else 0.0
                    if isinstance(v, str):
                        return 1.0 if v.strip().lower() == "true" else 0.0
                # Got a reply but could not parse a boolean out of it: treat that as
                # non-compliant rather than unjudged, since a judge that honours the
                # "return JSON only" instruction would never reach this branch.
                return 0.0
        except Exception as e:
            print(f"  [sysprompt] retry {attempt+1}/{MAX_RETRIES}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            await asyncio.sleep(_backoff(attempt))
    print(f"  [sysprompt] FAILED after {MAX_RETRIES} retries", file=sys.stderr)
    return None


async def call_judge(session: aiohttp.ClientSession, reply: str,
                     key_points: list) -> Optional[list]:
    """Call the judge and return the numbers of the criteria it found met. None when retries run out."""
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": build_judge_user_message(reply, key_points)},
        ],
        "temperature": JUDGE_TEMPERATURE,
        "max_tokens": JUDGE_MAX_TOKENS,
        "enable_thinking": False,
    }
    headers = {
        "Authorization": f"Bearer {JUDGE_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(
                JUDGE_API_URL, headers=headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:200]
                    # 4xx that retrying cannot fix (bad key, malformed request, wrong path). Retrying
                    # these MAX_RETRIES times burns hundreds of calls per training step for nothing.
                    if resp.status in (400, 401, 403, 404):
                        print(f"  [judge] HTTP {resp.status} is not retryable: {body}",
                              file=sys.stderr)
                        return None
                    print(f"  [judge] HTTP {resp.status} (attempt {attempt+1}/{MAX_RETRIES}): {body}",
                          file=sys.stderr)
                    await asyncio.sleep(_backoff(attempt))
                    continue
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    hits = json.loads(match.group()).get("hits", [])
                    # A "hits" value that is not a list degrades to empty.
                    return hits if isinstance(hits, list) else []
                return []
        except Exception as e:
            print(f"  [judge] retry {attempt+1}/{MAX_RETRIES}: {type(e).__name__}: {e}",
                  file=sys.stderr)
            await asyncio.sleep(_backoff(attempt))

    print(f"  [judge] FAILED after {MAX_RETRIES} retries", file=sys.stderr)
    return None
