"""Optional: wire reward.py into a GSPO trainer.

reward.py is the reward. This file is only the plumbing that hands a rollout engine's
completions to it, and adds the one term that needs a whole rollout group to exist. Nothing
here defines the metric; delete this file and evaluation is unaffected.

CONTRACT

    async def omni_basic_reward(args, sample)        -> float | dict   # one completion
    async def omni_basic_reward(args, samples: list) -> list           # one whole group

Point the trainer's custom-reward hook at `trainer_adapter.omni_basic_reward`. The group form is
what makes the word-efficiency term computable; see 5 below. A sample needs four attributes --
the raw completion, the rubric (`label` or `solution`), a per-prompt group index and an index
within the group -- plus, optionally, a status field marking a truncated generation.

Scoring itself lives in reward.py, which IS the definition of the metric: the judge
prompt, the tier gate, format_score, count_words, combine_score. The evaluation path imports
the same file. Keep one copy; two copies drift and then the two paths no longer measure the
same thing.

    score = rubric(hit_rate) + FORMAT_WEIGHT * format + WORD_EFF_WEIGHT * word_eff
                             + SYSPROMPT_WEIGHT * sysprompt

Template defaults 1 : 0.5 : 0 : 0, so the range is [0, 1.5]. Set FORMAT_WEIGHT=0 to get back
exactly score == hit_rate. The evaluation path deliberately reports the rubric term alone,
range [0, 1] -- so a training reward mean is NOT on the same scale as a benchmark number. That
split is intentional: word_eff is normalised inside a rollout group and does not exist outside
one, and keeping the benchmark at the pure rubric is what keeps numbers comparable across
reward designs.

--------------------------------------------------------------------------------------------
FIVE PROPERTIES OF THE ROLLOUT SETUP DRIVE THE SHAPE OF THIS FILE. Four are constraints; the
fifth is a free upgrade.

1. THE MODEL MAY EMIT THE '<think>' OPENER ITSELF, so it IS in the completion -- and
   format_score demands it is NOT. Which of the two happens is a property of the chat
   template: if its generation prompt ends on the assistant turn without a bare opener, the
   model volunteers '<think>' and the engine returns that text verbatim.

   format_score() at REQUIRE_THINK_CLOSE=1 requires ZERO openers, so an un-normalised
   completion would score format 0.0 on EVERY sample forever. And because the advantage uses a
   group baseline -- the group mean is subtracted -- a term constant across the group CANCELS -- the failure
   is not a wrong number in the loss, it is a 0.5-weighted term that silently never fires while
   the logs report `format=0.000`. So _normalize_completion strips the ONE leading opener
   before handing the text over, which keeps reward.py identical instead of forking a
   third parsing mode into it. A trainer whose decode step re-prepends the prefix needs the
   same stripper for the opposite reason. The first batch LOGS the observed tag shape, so the
   log is the evidence rather than this comment.

2. NOTHING MAY RAISE. An async custom reward is awaited in the caller's event loop, and
   anything that escapes it takes the rollout down. Every path funnels into the fallback
   instead; _score_one has a single exit.

3. JUDGE FAILURES BECOME A PER-PROCESS RUNNING MEAN of the rubric scores already collected,
   never a silent 0. A 0 is indistinguishable from a genuinely bad answer, and it would drag
   the group mean down and distort every advantage in that group.

4. THE JUDGE CONCURRENCY CEILING IS NOT PER-RANK HERE. A trainer that runs the reward once per
   torchrun rank multiplies JUDGE_MAX_CONCURRENCY by the number of ranks. This function is
   awaited in ONE process instead, gated by a single reward-side concurrency limit, so the
   effective ceiling is that limit times JUDGE_MAX_CONCURRENCY. Set the reward-side limit so
   the product stays inside what the judge endpoint tolerates. The semaphore INSIDE this file
   is still per call; see _score_batch.

5. THE UPGRADE: word_eff is actually computable here. A reward that only sees one rank's
   micro-batch sees a group of N split across ranks, and the term can never be computed. When
   the engine hands this function a WHOLE GROUP in one call and each sample carries a
   per-prompt group index, the grouping is READ rather than inferred from identical rubric
   strings. It stays weighted 0 by default, but the option is real.
"""
import asyncio
import json
import os
import random
import sys
import time
from collections import defaultdict

import aiohttp

# reward.py sits next to this file. Adding this directory ourselves means the trainer only has
# to make this module importable, not both.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reward import (  # noqa: E402
    FORMAT_WEIGHT,
    JUDGE_MAX_CONCURRENCY,
    JUDGE_MODEL,
    JUDGE_PROMPT_SHA,
    JUDGE_START_JITTER,
    REQUIRE_THINK_CLOSE,
    SYSPROMPT_WEIGHT,
    TARGET_MODEL_NAME,
    WORD_EFF_WEIGHT,
    call_judge,
    call_sysprompt_judge,
    combine_score,
    compute_tier_gated_score,
    count_words,
    format_score,
    hit_rate,
    identity_aliases,
    parse_key_points,
    require_judge_config,
    strip_thinking,
)

# Fail at IMPORT, not per sample: called inside the reward the RuntimeError would be caught by the
# catch-all and turned into a fallback score, which is the silent all-running-mean run this check
# exists to prevent. The judge settings must therefore already be in the rollout process's
# environment by the time the trainer imports this file -- export them in the launcher and, if the
# trainer runs workers under a runtime env, forward them into it.
require_judge_config()

# Group-relative word efficiency. See specific 5. Non-zero needs --group-rm so one call sees a group.
WORD_EFF_ON = WORD_EFF_WEIGHT != 0.0
# How many completions make a group. Trainers usually already export this as
# N_SAMPLES_PER_PROMPT, which is read as the default; WORD_EFF_GROUP_SIZE overrides it for a
# topology where the two stop being the same number.
WORD_EFF_GROUP_SIZE = int(os.environ.get("WORD_EFF_GROUP_SIZE")
                          or os.environ.get("N_SAMPLES_PER_PROMPT") or 0)
# What to hand back when the group carries no relative information (every member tied, or the group is
# not complete in this call). Any constant is equivalent for learning — the group baseline subtracts
# the group mean,
# so a term identical across the group cancels — and the midpoint keeps the logs readable.
WORD_EFF_NEUTRAL = 0.5

# Per-completion dump: one JSONL per training step, holding every TERM of the reward rather than
# only the final scalar. A trainer's own rollout log usually records prompt/response/reward/length,
# which is deliberately not a substitute -- it cannot tell you WHICH term moved.
ROLLOUT_DUMP_DIR = os.environ.get("ROLLOUT_DUMP_DIR", "")


# ⚠ SOME INFERENCE ENGINES RETURN THE STOP TOKEN IN THE TEXT, AND IT REACHES THE JUDGE.
# Whether it appears depends on how the engine detokenizes: with special tokens skipped it does not,
# without them it does. Measured on one such setup: `<|im_end|>` was present in 371/384 thinking
# completions and 400/400 instruct completions, and `strip_thinking()` keeps it, so 198 of 200
# sampled REPLIES handed to the judge ended in a literal `<|im_end|>`.
#
# Two measured consequences, one small and one exact:
#   * `count_words("<|im_end|>") == 2`, so every reply's word count was inflated by exactly 2. At a
#     non-zero WORD_EFF_WEIGHT that biases the term directly.
#   * the style criterion forbids "special symbols" verbatim. A/B on 24 style-failed completions,
#     same judge, same prompt: 0/24 compliant as-is vs 1/24 with the token stripped -- so it accounts
#     for roughly 4% of those failures, real but not the main cause.
#
# Stripped here, in the adapter, never in reward.py: that file is the shared definition of the
# metric and the evaluation path imports the same copy. Listed explicitly rather than regex-matching
# `<|...|>`, because a marker the MODEL emitted mid-reply is malformed output that must keep costing
# format and style score; only the engine's own terminator is noise.
_ENGINE_TOKENS = ("<|im_end|>", "<|endoftext|>")


def _strip_engine_tokens(text: str) -> str:
    """Remove the rollout engine's terminator tokens, which are transport, not content."""
    if not text:
        return text
    for tok in _ENGINE_TOKENS:
        text = text.replace(tok, "")
    return text


def _normalize_completion(text: str) -> str:
    """Present an engine completion in the shape reward.py's think branch expects.

    Drops the ONE leading '<think>' opener the MODEL emits and the engine's stop token, leaving
    `reasoning + '</think>' + reply` — what format_score(REQUIRE_THINK_CLOSE=1) grades. Deliberately
    narrow:

      * only a LEADING opener is removed, and only one. An opener appearing later in the text is the
        model starting a second reasoning block, which is malformed and must keep scoring format 0.
      * nothing else is touched, so a completion that has no opener at all passes through unchanged
        and both strip_thinking() and format_score() behave as they do in the other repos.

    Only applied when REQUIRE_THINK_CLOSE=1, i.e. when format_score is actually asking "is the opener
    absent". At 0 the opener is meaningful evidence and removing it would hide a malformed reply.
    """
    # The engine's stop token goes unconditionally: it is transport in both branches, and at
    # REQUIRE_THINK_CLOSE=0 (instruct) this is the ONLY normalisation that happens.
    text = _strip_engine_tokens(text)
    if not REQUIRE_THINK_CLOSE or not text:
        return text
    stripped = text.lstrip()
    if stripped.startswith("<think>"):
        return stripped[len("<think>"):].lstrip("\n")
    return text


def _word_eff_by_group(group_keys, rubrics, words):
    """Min-max the per-sample efficiency rubric/words INSIDE each complete group. -> one value each.

    raw efficiency = rubric / words(reply), normalised to [0,1] across the group: the question it
    answers is "of the N answers to this one prompt, which earned its rubric credit in the fewest
    words". Cross-prompt comparison is deliberately not attempted — a question that simply needs a
    longer answer would otherwise read as an inefficient one. The denominator is the REPLY, not the
    completion: this is a thinking checkpoint, and charging for the reasoning block would turn the term
    into a "reason less" penalty, which is the opposite of what the checkpoint is for.

    Grouping is READ from the per-prompt group index the data source assigns, not inferred from
    identical rubric strings -- inference merges two distinct prompts that happen to share a rubric.
    A group only counts as complete when it has exactly WORD_EFF_GROUP_SIZE members in this call,
    which requires the trainer to hand a whole group to one reward call.
    """
    if not WORD_EFF_ON:
        # Exactly 0.0, not neutral: at weight 0 the term is dropped, and reporting 0.5 in the dump
        # would suggest a value was computed. Same convention the other two repos follow.
        return [0.0] * len(group_keys)
    out = [WORD_EFF_NEUTRAL] * len(group_keys)
    if WORD_EFF_GROUP_SIZE < 2:
        return out

    groups = defaultdict(list)
    for i, key in enumerate(group_keys):
        groups[key].append(i)

    complete = 0
    for idxs in groups.values():
        if len(idxs) != WORD_EFF_GROUP_SIZE:
            continue
        complete += 1
        raw = [rubrics[i] / max(words[i], 1) for i in idxs]
        lo, hi = min(raw), max(raw)
        span = hi - lo
        for i, value in zip(idxs, raw):
            out[i] = WORD_EFF_NEUTRAL if span <= 0.0 else (value - lo) / span
    if complete == 0 and not _State.warned_word_eff:
        print("[reward] WORD_EFF_WEIGHT=%s but no complete group of %d reached this call (%d "
              "completion(s), %d distinct group_index): the term needs --group-rm so one call sees a "
              "whole group. Falling back to the neutral constant, which cancels against the group baseline."
              % (WORD_EFF_WEIGHT, WORD_EFF_GROUP_SIZE, len(group_keys), len(groups)),
              file=sys.stderr, flush=True)
        _State.warned_word_eff = True
    return out


class _State:
    """Per-process counters. A module-level class rather than globals so the names stay grepped."""

    total_calls = 0
    total_judge_failures = 0
    batch_count = 0
    # Completions dumped so far. The training step is derived from this and the batch
    # geometry, because the Sample does not carry a rollout_id — see _dump().
    dumped = 0
    logged_config = False
    warned_word_eff = False
    # Running mean of the RUBRIC only, not of the final score: it substitutes for a judge call that
    # failed, and the judge is the only thing that produces the rubric. format and word_eff are
    # computed locally and are available even when the endpoint is down, so they are added on top of
    # the substituted rubric rather than being guessed along with it.
    rubric_sum = 0.0
    rubric_count = 0

    @classmethod
    def fallback_rubric(cls):
        """Running mean of this process's successful rubric scores; 0.0 before any success."""
        return cls.rubric_sum / cls.rubric_count if cls.rubric_count else 0.0


def _log_config_once(completions):
    if _State.logged_config:
        return
    n_open = sum(1 for c in completions if "<think>" in (c or ""))
    n_close = sum(1 for c in completions if "</think>" in (c or ""))
    print(f"[reward] target={TARGET_MODEL_NAME} aliases={identity_aliases()} "
          f"judge_model={JUDGE_MODEL} judge_prompt_sha={JUDGE_PROMPT_SHA} "
          f"scoring=tier_gated(tier-0 hard gate) "
          f"require_think_close={REQUIRE_THINK_CLOSE} "
          f"max_concurrency={JUDGE_MAX_CONCURRENCY}/call "
          f"reward=rubric+{FORMAT_WEIGHT}*format+{WORD_EFF_WEIGHT}*word_eff"
          f"{f'+{SYSPROMPT_WEIGHT}*sysprompt' if SYSPROMPT_WEIGHT else ''} "
          f"word_eff_group={WORD_EFF_GROUP_SIZE if WORD_EFF_ON else 'off'}", flush=True)
    # Constraint 1: this line is the EVIDENCE that _normalize_completion is doing the right thing.
    # '<think>' in N/N means the model is emitting the opener as expected and the normaliser is
    # load-bearing; 0/N would mean it is a no-op and format_score needs no help. On this stack N/N is
    # the expected reading — if it says 0/N, something is putting the opener in the prompt and this
    # file's constraint 1 needs re-checking.
    print(f"[reward] completion tag shape: '<think>' in {n_open}/{len(completions)}, "
          f"'</think>' in {n_close}/{len(completions)} "
          f"(normaliser {'active' if n_open else 'NO-OP — see constraint 1'})", flush=True)
    _State.logged_config = True


async def _score_one(session, sem, completion, label, truncated):
    """One completion -> its reward components. Single exit; never raises."""
    async with sem:
        text = _normalize_completion(completion or "")
        # The RAW (normalised) completion, not the reply: the format term grades where the tags are.
        fmt = format_score(text)
        reply = strip_thinking(text)
        rubric, judge_ok, reply_ok, scoring = 0.0, 1, 1, None

        if not reply:
            # Judge-independent zero: either the reasoning block ran out of the response budget before
            # any reply, or the reply really was blank. reply_ok separates "nothing was answered" from
            # "something bad was answered" — without it a collapse to 0 looks identical in both cases.
            # `truncated` is the engine telling us which, from a sample status set when the
            # generation stopped on the length limit rather than on a stop token.
            reply_ok = 0
        else:
            key_points = parse_key_points(label)
            if not key_points:
                # Data defect, not a model failure. Scoring it 0 would invent a penalty, so fall
                # back and mark the sample untrusted -- the same treatment eval/score.py gives it,
                # where such a sample is excluded from the mean.
                rubric, judge_ok = _State.fallback_rubric(), 0
            else:
                # Jitter INSIDE the semaphore is fine here because the semaphore is this call's own
                # and the endpoint's limit is on concurrent requests, not on held slots. A setup whose
                # semaphore is shared across a whole micro-batch has to jitter outside it instead; the
                # shapes differ, the intent is the same -- do not start N requests on the same
                # millisecond.
                if JUDGE_START_JITTER:
                    await asyncio.sleep(random.uniform(0, JUDGE_START_JITTER))
                hits = await call_judge(session, reply, key_points)
                if hits is None:
                    rubric, judge_ok = _State.fallback_rubric(), 0
                else:
                    scoring = compute_tier_gated_score(key_points, hits)
                    rubric = hit_rate(scoring)
                    _State.rubric_sum += rubric
                    _State.rubric_count += 1

        # sysprompt: only asked for when it is actually weighted, so the default config pays no extra
        # judge call. An empty reply scores 0 without a call — there is no style to obey.
        sysprompt, sysprompt_ok = 0.0, 1
        if SYSPROMPT_WEIGHT and reply:
            if JUDGE_START_JITTER:
                await asyncio.sleep(random.uniform(0, JUDGE_START_JITTER))
            v = await call_sysprompt_judge(session, reply)
            if v is None:
                # Do NOT substitute 0.0 — that reads as "violated the style" and would silently lower
                # the reward whenever the endpoint throttles. Neutral-but-flagged is the honest
                # choice, and sysprompt_ok keeps the rate visible in the dump.
                sysprompt, sysprompt_ok = 0.0, 0
            else:
                sysprompt = v

        return {"rubric": rubric, "fmt": fmt, "words": count_words(reply), "judge_ok": judge_ok,
                "reply_ok": reply_ok, "scoring": scoring, "sysprompt": sysprompt,
                "sysprompt_ok": sysprompt_ok, "reply": reply, "truncated": int(bool(truncated))}


async def _score_batch(completions, labels, group_keys, truncateds, indices=None, args=None,
                       prompts=None):
    """Score N completions -> N floats, plus the parts for the dump. Never raises."""
    _State.batch_count += 1
    step = _State.batch_count
    _log_config_once(completions)

    # Per-CALL semaphore. Hosted judge endpoints answer concurrent bursts with HTTP 429 and ask
    # clients to spread requests out, and some answer HTTP 500 with an out-of-memory message when
    # pushed -- a ceiling on the SERVICE, not on local GPUs. The AGGREGATE is this times the
    # trainer's reward-side concurrency limit; set that limit so the product stays inside what the
    # endpoint tolerates -- see constraint 4.
    sem = asyncio.Semaphore(JUDGE_MAX_CONCURRENCY)

    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        scored = await asyncio.gather(
            *[_score_one(session, sem, c, l, t)
              for c, l, t in zip(completions, labels, truncateds)],
            return_exceptions=True)

    # Constraint 2: a stray exception must not cancel the rest and must not take the rollout down.
    parts = []
    for r in scored:
        if isinstance(r, BaseException):
            print(f"[reward] unexpected {type(r).__name__}: {r} -> fallback",
                  file=sys.stderr, flush=True)
            parts.append({"rubric": _State.fallback_rubric(), "fmt": 0.0, "words": 0,
                          "judge_ok": 0, "reply_ok": 1, "scoring": None,
                          "sysprompt": 0.0, "sysprompt_ok": 1, "reply": "", "truncated": 0})
        else:
            parts.append(r)

    word_eff = _word_eff_by_group(group_keys,
                                 [p["rubric"] for p in parts],
                                 [p["words"] for p in parts])
    rewards = [combine_score(p["rubric"], p["fmt"], we, p["sysprompt"])
               for p, we in zip(parts, word_eff)]

    judge_failures = sum(1 for p in parts if not p["judge_ok"])
    empty_replies = sum(1 for p in parts if not p["reply_ok"])
    truncs = sum(p["truncated"] for p in parts)
    _State.total_calls += len(parts)
    _State.total_judge_failures += judge_failures

    mean = sum(rewards) / len(rewards) if rewards else 0.0
    rubric_mean = (sum(p["rubric"] for p in parts) / len(parts)) if parts else 0.0
    fmt_mean = (sum(p["fmt"] for p in parts) / len(parts)) if parts else 0.0
    msg = (f"[reward] CALL {step}: n={len(parts)} score={mean:.3f} rubric={rubric_mean:.3f} "
           f"format={fmt_mean:.3f} empty_reply={empty_replies} truncated={truncs} "
           f"judge_failures={judge_failures} "
           f"(total {_State.total_judge_failures}/{_State.total_calls}) "
           f"{time.time() - t0:.1f}s")
    if judge_failures:
        msg += f" [substituted running mean {_State.fallback_rubric():.3f}]"
    print(msg, flush=True)

    _dump(completions, labels, parts, word_eff, rewards,
          group_keys=group_keys, indices=indices, args=args, prompts=prompts)
    return rewards, parts, word_eff


def _training_step(args, dumped_before, n_in_call):
    """Which TRAINING step the completions in this call belong to.

    Derived, because it usually cannot be read: a sample carries its group index and its index
    within the group, but not the training step -- the step is known to the trainer's dump writer,
    not to the reward. What the reward does have is `args`, so the geometry is available:

        completions per step == global_batch_size          (one call is one completion at GROUP_RM off,
                                                            one prompt group at GROUP_RM on — either way
                                                            the COMPLETION count per step is the same)

    Falls back to 0 when global_batch_size is missing or nonsensical, which puts everything in one file
    rather than inventing a step number.
    """
    try:
        per_step = int(getattr(args, "global_batch_size", 0) or 0)
    except (TypeError, ValueError):
        per_step = 0
    if per_step <= 0:
        return 0
    return dumped_before // per_step


def _dump(completions, labels, parts, word_eff, rewards, group_keys=None, indices=None, args=None,
          prompts=None):
    """Append this call's completions to output/rollouts/<training step>.jsonl.

    One record per completion, carrying every term separately (`rubric_score`, `format_score`,
    `word_eff`, `sysprompt_score`) next to the combined `score`. Appended, not written: many calls
    make up one training step and their samples are disjoint. Best-effort -- a dump failure must
    never affect training.

    ⚠ THE FILENAME IS THE STEP LABEL FOR THE WHOLE ROLLOUT-SCORING CURVE, so getting it wrong gives a
    wrong curve rather than an untidy directory. Naming the file after a per-call counter looks right
    and is not: when one call is one COMPLETION, a single real step becomes dozens of one-sample
    "steps", and the directory grows by that factor (measured: 6144 files by step 47, heavy in inodes
    long before it is heavy in bytes). Hence the derived step, one file per real step.

    ⚠ THE DERIVED STEP RESETS IF THE ROLLOUT PROCESS RESTARTS, because the counter is per-process.
    Later steps then append to files that already hold earlier ones, so the effect is a MERGED step
    rather than a lost one -- appending is deliberate. `ROLLOUT_DUMP_STEP` overrides the name outright
    for a caller that knows the step.

    Each record also carries the group index and the index within the group, so this dump can be
    JOINED to the trainer's own rollout log -- which holds the complementary half (prompt, rubric,
    lengths, final reward) and often shares no field names with this one.
    """
    if not ROLLOUT_DUMP_DIR:
        return
    try:
        os.makedirs(ROLLOUT_DUMP_DIR, exist_ok=True)
        step = _training_step(args, _State.dumped, len(completions))
        _State.dumped += len(completions)
        name = os.environ.get("ROLLOUT_DUMP_STEP") or str(step)
        path = os.path.join(ROLLOUT_DUMP_DIR, "%s.jsonl" % name)
        gks = list(group_keys or [None] * len(completions))
        idxs = list(indices or [None] * len(completions))
        prs = list(prompts or [None] * len(completions))
        with open(path, "a", encoding="utf-8") as f:
            for k, (completion, label, p, we, score) in enumerate(
                    zip(completions, labels, parts, word_eff, rewards)):
                sc = p["scoring"]
                pr = prs[k] if k < len(prs) else None
                f.write(json.dumps({
                    "step": step,
                    "group_index": gks[k] if k < len(gks) else None,
                    "index": idxs[k] if k < len(idxs) else None,
                    "input": pr if isinstance(pr, str) else (
                        json.dumps(pr, ensure_ascii=False) if pr is not None else None),
                    "output": completion,
                    "gts": label if isinstance(label, str) else json.dumps(label, ensure_ascii=False),
                    "score": round(float(score), 6),
                    "rubric_score": round(float(p["rubric"]), 6),
                    "format_score": float(p["fmt"]),
                    "word_eff": round(float(we), 6),
                    "sysprompt_score": float(p["sysprompt"]),
                    "sysprompt_ok": float(p["sysprompt_ok"]),
                    "reply_words": float(p["words"]),
                    "judge_ok": float(p["judge_ok"]),
                    "reply_ok": float(p["reply_ok"]),
                    "truncated": float(p["truncated"]),
                    "hit_count": float(sc["score"]) if sc else 0.0,
                    "kp_total": float(sc["scorable_keypoints"]) if sc else 0.0,
                    "max_tier": float(sc["max_unlocked_tier"]) if sc else 0.0,
                    "gate_passed": float(bool(sc["gate_passed"])) if sc else 0.0,
                }, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001 — constraint 2
        print(f"[reward] rollout dump failed ({type(e).__name__}: {e}); training unaffected",
              file=sys.stderr, flush=True)


def _shape(value, reward_key):
    """A float, or {reward_key: float} when --reward-key is set.

    A trainer that reads the reward as `sample.reward[args.reward_key]` needs the dict form to
    contain exactly that key; it may carry anything else alongside.
    """
    return {reward_key: float(value)} if reward_key else float(value)


async def omni_basic_reward(args, sample, **kwargs):
    """THE ENTRY POINT. Mirrors its input shape: one Sample -> one reward, a list -> a list.

    Trainers call it both ways depending on whether group-level rewards are enabled, and returning
    the wrong shape is a `zip(samples, rewards)` that silently drops samples rather than an error --
    so the shape is derived from the argument, never assumed.
    """
    reward_key = getattr(args, "reward_key", None)
    single = not isinstance(sample, (list, tuple))
    samples = [sample] if single else list(sample)

    try:
        completions = [getattr(s, "response", "") or "" for s in samples]
        labels = [getattr(s, "label", None) for s in samples]
        # group_index is exact here; fall back to the sample's own id so a missing one makes the group
        # incomplete (-> neutral word_eff) rather than merging every sample into one group.
        group_keys = [getattr(s, "group_index", None)
                      if getattr(s, "group_index", None) is not None else ("i%d" % i)
                      for i, s in enumerate(samples)]
        truncateds = [str(getattr(s, "status", "")).endswith("TRUNCATED") for s in samples]
        # The index within the group. Carried through purely so this dump and the trainer's own can
        # be JOINED -- see _dump().
        indices = [getattr(s, "index", None) for s in samples]
        # ⚠ `input` AND `step` MAKE THE DUMP SELF-CONTAINED. Without the prompt, recovering what a
        # dumped completion was answering means joining to the trainer's own rollout log; the sample
        # already carries it, so there is no reason to make the reader join. `step` is written even
        # though the filename encodes it, because that filename is derived from a per-process counter
        # that RESETS if the rollout process restarts -- see _dump(). A field inside the record
        # survives that.
        prompts = [getattr(s, "prompt", None) for s in samples]

        rewards, _parts, _we = await _score_batch(completions, labels, group_keys, truncateds,
                                                  indices=indices, args=args, prompts=prompts)
    except Exception as e:  # noqa: BLE001 — constraint 2: the rollout must survive anything here
        print(f"[reward] FATAL {type(e).__name__}: {e} -> every sample gets the running mean",
              file=sys.stderr, flush=True)
        fb = _State.fallback_rubric()
        rewards = [fb] * len(samples)

    shaped = [_shape(r, reward_key) for r in rewards]
    return shaped[0] if single else shaped


if __name__ == "__main__":
    # Self-check, runnable without a trainer, without a training run and without the dataset:
    #     export JUDGE_API_BASE=... JUDGE_API_KEY=... JUDGE_MODEL=...
    #     python3 trainer_adapter.py
    # The offline invariants (count_words, format_score, combine_score, the completion normaliser,
    # the word_eff grouping, the return SHAPE) must fail in milliseconds rather than after four API
    # calls, and the judge leg must prove the judge is actually GRADING rather than the pipeline
    # merely running.
    #
    # The fixture is literal, not read from the dataset. A positive control has to be a reply that
    # genuinely satisfies its rubric, and rubric text is written in the third person ABOUT a reply
    # ("The response states that ..."), so feeding the criteria back as if they were an answer
    # produces a meta-description that a correctly-working judge REFUSES to credit -- a self-check
    # that fails when everything is fine. Four criteria and one reply that answers all four keep the
    # positive control honest.
    FIXTURE_RUBRIC = [
        {"tier": 0, "point": "The response is in English, consistent with the user's language."},
        {"tier": 1, "point": "The response confirms it can see the mug on the desk."},
        {"tier": 2, "point": "The response says the mug is red."},
        {"tier": 3, "point": "The response offers to help with something related."},
    ]
    GOOD_REPLY = ("Yes, I can see the mug on your desk, and it is the red one. "
                  "Want me to keep an eye on it while you step away?")
    labels = [json.dumps({"key_points": FIXTURE_RUBRIC}, ensure_ascii=False)] * 3

    class _FakeSample:
        """The three fields omni_basic_reward reads, plus the two it tolerates missing."""

        def __init__(self, response, label, group_index=None, status=""):
            self.response = response
            self.label = label
            self.group_index = group_index
            self.status = status

    class _FakeArgs:
        reward_key = None

    def completion(reasoning, reply):
        """Shape a completion the way a rollout produces it.

        At REQUIRE_THINK_CLOSE=1 the MODEL emits the opener (see constraint 1), so the reward receives
        `'<think>\\n' + reasoning + '</think>' + reply` — opener INCLUDED. That is precisely what
        _normalize_completion exists to undo, so the check must include it.
        """
        if REQUIRE_THINK_CLOSE:
            return "<think>\n%s\n</think>\n\n%s" % (reasoning, reply)
        return reply

    cases = [
        ("good reply", completion("Let me look at the desk.", GOOD_REPLY), 1),
        ("off-topic reply", completion("Hmm, no idea.", "I like golf."), 1),
        ("empty reply", completion("I have nothing to add.", ""), 0),
    ]
    if REQUIRE_THINK_CLOSE:
        # Reasoning that never closed: the response budget ran out mid-trace. Must score 0 and must
        # never reach the judge — grading a raw reasoning trace is the bug strip_thinking prevents.
        cases.append(("truncated reasoning",
                      "<think>\nThe user is asking about the video, so first I should", 0))

    def check_offline():
        # ⚠ THE NORMALISER'S CORRECT BEHAVIOUR FLIPS WITH REQUIRE_THINK_CLOSE, so both policies are
        # asserted rather than one. _normalize_completion() returns its input unchanged at 0 by its own
        # first line — that is the documented contract, not a shortcut. Keeping the two stripping
        # assertions inside this branch matters: outside it they fail on every REQUIRE_THINK_CLOSE=0
        # setup, and the failure looks like a judge problem rather than a self-check problem.
        if REQUIRE_THINK_CLOSE:
            # The model volunteers the opener (constraint 1), so the ONE leading opener goes...
            assert _normalize_completion("<think>\na\n</think>\nb") == "a\n</think>\nb"
            assert _normalize_completion("  <think>a</think>b") == "a</think>b"
            assert "<think>" not in _normalize_completion("<think>\nx\n</think>\ny")
            # ...and a LATER one stays: it is the model starting a second reasoning block, which is
            # malformed and must keep scoring format 0.
            assert _normalize_completion("<think>\na<think>b\n</think>\nc") == "a<think>b\n</think>\nc"
        else:
            # A non-thinking checkpoint emits no opener, so this is a NO-OP — and it has to be. Stripping
            # here would delete the evidence that a reply is malformed and hand format_score a
            # well-formed-looking string, turning a 0 into a 1.
            assert _normalize_completion("<think>\na\n</think>\nb") == "<think>\na\n</think>\nb"
            assert _normalize_completion("  <think>a</think>b") == "  <think>a</think>b"
        # True under either policy: nothing to strip, nothing stripped.
        assert _normalize_completion("a</think>b") == "a</think>b"
        assert _normalize_completion("") == ""

        # count_words: the property that matters is that the same answer written in Chinese and in
        # English lands in the same order of magnitude. If it did not, word_eff would quietly become a
        # language detector and reward answering in whichever language counts cheaper.
        assert count_words("") == 0
        assert count_words("。，！？ …") == 0, "punctuation is not a word"
        assert count_words("Hello, world!") == 2
        assert count_words("don't stop") == 2, "an apostrophe does not split a word"
        assert count_words("state-of-the-art") == 1
        assert count_words("你好，世界！") == 4, "one hanzi is one word"
        assert count_words("中文 mixed 混排 123") == 6

        # format_score, on the NORMALISED text — which is what _score_one grades. These are the
        # assertions that would fail if constraint 1 were wrong.
        if REQUIRE_THINK_CLOSE:
            assert format_score(_normalize_completion(completion("r", "the reply"))) == 1.0
            assert format_score(completion("r", "the reply")) == 0.0, \
                "the RAW completion must score 0 — that is WHY the normaliser is load-bearing"
            assert format_score(_normalize_completion("<think>\nnever closed")) == 0.0
            assert format_score(_normalize_completion(
                "<think>\na\n</think>\nb\n</think>\nc")) == 0.0, "two closers is malformed"
            assert format_score(_normalize_completion(
                "<think>\na<think>again\n</think>\nreply")) == 0.0, "a second opener is malformed"
            assert format_score(_normalize_completion(completion("r", "   "))) == 0.0, \
                "closed but no reply"

        # The advertised 1 : FORMAT : WORD_EFF : SYSPROMPT ratio, asserted rather than assumed.
        assert abs(combine_score(1.0, 0.0, 0.0, 0.0) - 1.0) < 1e-9
        assert abs(combine_score(0.0, 1.0, 0.0, 0.0) - FORMAT_WEIGHT) < 1e-9
        assert abs(combine_score(0.0, 0.0, 1.0, 0.0) - WORD_EFF_WEIGHT) < 1e-9
        assert abs(combine_score(0.0, 0.0, 0.0, 1.0) - SYSPROMPT_WEIGHT) < 1e-9

        # word_eff grouping, now keyed on group_index: complete groups normalise, incomplete ones fall
        # back, weight 0 drops it.
        #
        # warned_word_eff is pinned True for the duration: the incomplete-group assertion below
        # deliberately takes the fallback path, whose one-time message interpolates WORD_EFF_WEIGHT —
        # and under this globals() patch that reads the REAL weight (0.0) while WORD_EFF_ON says True,
        # so it would print the self-contradictory "WORD_EFF_WEIGHT=0.0 but no complete group". The
        # grouping logic is still fully asserted; only the print is muted.
        saved = (globals()["WORD_EFF_ON"], globals()["WORD_EFF_GROUP_SIZE"],
                 _State.warned_word_eff)
        try:
            globals()["WORD_EFF_ON"], globals()["WORD_EFF_GROUP_SIZE"] = True, 4
            _State.warned_word_eff = True
            # Integral rubrics keep the expected min-max EXACT in binary floating point; 0.4/0.8/0.2
            # would land on 0.7500000000000001 and the assertion would be about float error.
            got = _word_eff_by_group([7, 7, 7, 7], [10.0, 4.0, 8.0, 2.0], [1, 1, 1, 1])
            assert got == [1.0, 0.25, 0.75, 0.0], got
            assert _word_eff_by_group([7] * 4, [0.5] * 4, [1] * 4) == [WORD_EFF_NEUTRAL] * 4, \
                "an all-tied group carries no relative information"
            assert _word_eff_by_group([7, 8], [1.0, 0.0], [1, 1]) == [WORD_EFF_NEUTRAL] * 2, \
                "two different group_index values are not one group of 2"
            globals()["WORD_EFF_ON"] = False
            assert _word_eff_by_group([7] * 4, [1.0, 0.4, 0.8, 0.2], [1] * 4) == [0.0] * 4, \
                "at weight 0 the term must be exactly 0.0, not the neutral constant"
        finally:
            (globals()["WORD_EFF_ON"], globals()["WORD_EFF_GROUP_SIZE"],
             _State.warned_word_eff) = saved

    async def check_shape():
        """The return shape mirrors the input, and honours --reward-key. See omni_basic_reward."""
        args = _FakeArgs()
        one = _FakeSample(completion("r", "a reply"), labels[0], group_index=0)
        r = await omni_basic_reward(args, one)
        assert isinstance(r, float), "a single Sample must return a float, got %r" % type(r)

        many = [_FakeSample(completion("r", "a reply"), labels[0], group_index=0) for _ in range(2)]
        rs = await omni_basic_reward(args, many)
        assert isinstance(rs, list) and len(rs) == 2, "a list must return a same-length list"
        assert all(isinstance(x, float) for x in rs)

        class _KeyedArgs:
            reward_key = "score"
        rd = await omni_basic_reward(_KeyedArgs(), one)
        assert isinstance(rd, dict) and "score" in rd, \
            "--reward-key must produce {key: float}, got %r" % (rd,)
        print("OK: return shape mirrors input (float / list / {reward_key: float}).")

    async def main():
        check_offline()
        print("OK: offline invariants hold (normaliser, count_words, format_score, combine_score, "
              "word_eff grouping by group_index).")

        args = _FakeArgs()
        samples = [_FakeSample(text, lab, group_index=i)
                   for i, ((_label, text, _ok), lab) in enumerate(zip(cases, labels))]
        scores = await omni_basic_reward(args, samples)
        for (label, _text, expect_reply_ok), score in zip(cases, scores):
            print("  %-20s score=%.4f" % (label, score))
        # An empty/truncated reply must score 0 on both judged terms, and the good reply must
        # beat the off-topic one. The TOTAL is not asserted to be 0 for the no-reply cases — at a non-zero
        # WORD_EFF_WEIGHT the neutral constant contributes, which is correct and cancels in the
        # advantage.
        assert scores[0] > scores[1], (
            "the good reply (%.3f) did not outscore the off-topic one (%.3f): the judge is not "
            "grading, or the normaliser/strip_thinking is eating the reply" % (scores[0], scores[1]))
        for (label, _t, expect), score in zip(cases, scores):
            if expect == 0:
                assert score <= WORD_EFF_WEIGHT * WORD_EFF_NEUTRAL + 1e-9, \
                    "%s: a completion with no reply must score rubric=format=0, got %.3f" % (
                        label, score)
        print("OK: judge graded (good %.3f > off-topic %.3f), %d no-reply case(s) at 0, "
              "reward = rubric + %s*format + %s*word_eff + %s*sysprompt"
              % (scores[0], scores[1], sum(1 for _l, _t, e in cases if e == 0),
                 FORMAT_WEIGHT, WORD_EFF_WEIGHT, SYSPROMPT_WEIGHT))
        await check_shape()

    asyncio.run(main())
