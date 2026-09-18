# OmniVChat-RL reward

`reward.py` is the reward. Everything else here is optional.

- **`reward.py`** — the reward, and the only file that defines it: the judge prompt, the tier
  gate, `format_score`, `count_words`, `combine_score`, the style yardstick. Framework
  independent. `../eval/score.py` imports this same file, which is why the benchmark and the
  training reward cannot drift apart. If you fork it, both paths move together — that is the
  point of there being one copy.

- **`trainer_adapter.py`** — optional. Plumbing for wiring `reward.py` into a GSPO trainer:
  point the trainer's custom-reward hook at `trainer_adapter.omni_basic_reward` and it takes one
  completion or a whole rollout group and mirrors that shape on the way out. It also implements
  the one reward term that cannot exist for a single sample — group-relative efficiency — and
  writes a per-completion diagnostic dump. It defines no part of the metric, and deleting it
  changes nothing about evaluation. Its docstring lists the five properties of a rollout setup
  that shape it.

- **`prompts/`** — the judge prompts, read at import and fingerprinted into `JUDGE_PROMPT_SHA`,
  so an edit made between two launches of one experiment is findable afterwards.

Run `python3 trainer_adapter.py` for the self-check: offline invariants first (word counting,
format scoring, the weight ratio, group normalisation, the return shape), then four real judge
calls proving the judge grades a good reply above an off-topic one. It needs no trainer, no
training run and no dataset — the fixture is in the file.

## The reward

```
R(y) = r(y) + FORMAT_WEIGHT * f(y) + WORD_EFF_WEIGHT * e(y) + SYSPROMPT_WEIGHT * s(y)
```

| term | what it is | default weight |
|---|---|--:|
| `r(y)` | tier-gated rubric score, `[0, 1]` | 1 |
| `f(y)` | completion is structurally valid, `{0, 1}` | 0.5 |
| `e(y)` | reply efficiency, normalised **within the rollout group**, `[0, 1]` | 0 |
| `s(y)` | reply obeys the style guidelines, `{0, 1}` | 0 |

Range `[0, 1.5]` at the defaults. `FORMAT_WEIGHT=0` reduces it exactly to `r(y)`.

`e(y)` and `s(y)` are off by default because each costs something: `e(y)` needs one reward call
to see a whole group, `s(y)` adds one judge call per sample. Both are real, not aspirational —
switch them on with the weights.

`e(y)` is the one term that cannot be computed outside training. Raw efficiency is
`r(y) / words(reply)`; the term is that quantity min-max normalised across the N completions of
the **same** prompt, so it asks which of N answers earned its credit in the fewest words.
Comparing across prompts is deliberately avoided — a question that simply needs a longer answer
would otherwise read as an inefficient one. `../eval/score.py` reports the absolute version
instead and labels it as such.

## Configuration

Judge, via environment — any OpenAI-compatible chat-completions service:

```bash
export JUDGE_API_BASE=https://your-endpoint.example/v1
export JUDGE_API_KEY=...
export JUDGE_MODEL=<the judge model served there>
```

`JUDGE_API_URL` overrides the assembled URL for a service that does not use the `/v1` layout.
`JUDGE_TEMPERATURE` defaults to 0.1, `JUDGE_MAX_CONCURRENCY` to 4 per call — the aggregate is
that times your trainer's reward-side concurrency limit, so set the latter with the judge
endpoint's tolerance in mind.

Model under test: `TARGET_MODEL_NAME` and `TARGET_IDENTITY_NAMES` (comma-separated). The
MSA-IC rubrics ask the model to state its own name, and they score 0 against the placeholder
default.

Completion parsing: `REQUIRE_THINK_CLOSE` is `0` for an instruct checkpoint and `1` for a
thinking checkpoint whose chat template opens `<think>` in the prompt. Setting it wrong grades
a truncated reasoning trace as if it were a reply.

Diagnostics: `ROLLOUT_DUMP_DIR` writes one JSONL per training step with every term separately,
which is what tells you *which* term moved when the total does.
