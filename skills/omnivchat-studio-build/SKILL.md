---
name: omnivchat-studio-build
description: Build, maintain, or refactor OmniVChat-Studio for audio-visual dialogue synthesis, implementing its Director, Validator, Renderer, Reviewer, and single-turn and multi-turn subsystems. Use for framework implementation, module fixes, orchestration changes, and compatibility work; use the subcategory workflow for routine extensions to an existing framework.
---

# OmniVChat-Studio: Build and Maintain the Framework

Implement or maintain the framework that turns a text corpus and a Subcategory
Configuration into synchronized user audio and video, a text Reference Reply,
and an instance-level Rubric.

## Names and References

Use **OmniVChat-Studio**, its four roles (**Director, Validator, Renderer,
Reviewer**), and its **Single-turn Subsystem** and **Multi-turn Subsystem**.
Module names follow Figure 2, Section 2, and Appendix C of the
[OmniVChat paper](https://arxiv.org/abs/2609.21465).
Read [architecture.md](references/architecture.md) for names, responsibilities,
control flow, and data contracts.

The paper defines public terminology; the target code defines existing interfaces
and behavior. For maintenance, also read [legacy-map.md](references/legacy-map.md).
Use a mapping between display names and code paths. A naming cleanup alone should
not move historical data, change stable identifiers, or break resumability.

## Establish the Scope

Infer the target directory, existing implementation or new project, affected
subsystems, and expected behavior from the request and workspace. Read the target
repository's instructions, relevant README, and working-tree status. Clarify only
missing information that materially affects the task.

- **New project:** establish the contracts and role interfaces below, exercise an
  offline workflow, then connect backends supplied by the deployment environment.
- **Maintenance or refactoring:** reproduce the problem and identify affected
  modules, state, and consumers. Check each subsystem when a change crosses both.

Distribute architecture, data contracts, abstract backend interfaces, configuration
rules, and self-contained examples. Service integration belongs to the operator's
environment. Keep credentials, local runtime configuration, raw service logs,
deployment details, and unpublished provider information out of distributed files.

## Implementation Workflow

1. **Define contracts.** Separate sampled attributes, plans, scripts, media,
   evidence, replies, rubrics, and run state. Give resumable tasks stable IDs.
   Specify where single-turn results and final-turn annotations live. Update
   readers, writers, validators, and compatibility handling together.
2. **Separate the four roles.** The Director plans and decides; the Validator
   performs deterministic checks; the Renderer synthesizes synchronized media;
   the Reviewer provides captions, quality observations, and media-grounded Q-A.
   Replaceable backends and offline doubles should represent success, failure,
   and inconsistent evidence.
3. **Implement both paths.** Single-turn orchestration includes Script Generation,
   Validation, Script Review, Video Plan, and Reply Plan. Multi-turn orchestration
   has separate Plan, Plan Review, Prompt Generation, Prompt Review, final-turn
   Final-Reply Plan, and whole-dialogue Validation. Define each module's inputs,
   outputs, normal transitions, conditional branches, retry counters, and exits.
4. **Connect configuration.** Subcategory Configuration supplies dimensions,
   distributions, preferences, and validation rules. Inject the same rules into
   generation and validation. Prefer subcategory templates, with shared fallbacks.
   Confirm the actual variables supplied to each template.
5. **Implement recovery.** Persist sampled attributes, plan versions, completed
   turns, media references, and rework reasons. Re-seed preserves the script;
   Re-write returns to its drafting stage; Update Plan restarts at the earliest
   affected turn. Give every loop an observable budget and stopping condition.
6. **Check completion.** Export completed results only when media is usable,
   required replies exist, and rubric structure is valid. Record failures
   separately. Distinguish successful template rendering, simulated media, and
   acceptance based on real audio and video.

An optional layout for a new project follows. Preserve an existing project's
structure when maintaining it.

```text
omnivchat_studio/
  agents/                 # Director, Validator, Renderer, Reviewer
  single_turn/            # Single-turn orchestration and state
  multi_turn/             # Multi-turn orchestration and state
  sampling/               # Attribute & Corpus Sampling
  schemas/                # Data and decision contracts
  templates/              # Shared prompt templates
  subcategories/          # Subcategory configurations and templates
  media/                  # Media checks, references, derived assets
  persistence/            # Checkpoints and output writers
tests/
  fixtures/               # Self-contained inputs and offline doubles
```

## Behavioral Invariants

- Keep sampled attributes fixed through retries, rewrites, and plan updates.
  Resuming a batch must not replace difficult instances with new assignments.
- The Reviewer reports observations; the Director accepts or requests rework.
  Ground replies in confirmed media evidence when a script and rendering disagree.
- Validation Review may resolve semantic false positives. It cannot turn broken
  structure, missing media, or incomplete output into a completed result.
- Multi-turn references come from completed earlier turns. Invalidate dependent
  assets when redoing a turn, while retaining valid earlier results. Check a
  character's appearance and voice separately.
- Generate a Reference Reply for every intermediate turn and a Rubric for the
  final turn. Benchmark context alternates user clips and fixed intermediate
  replies, ending with the final user clip. Exclude the final reference answer
  and Rubric from model input; generate and score only the final reply.
- Tier 0 is the sole language prerequisite and earns no credit. A later tier
  contributes only after every earlier tier is fully satisfied; individual
  criteria within the eligible tier can earn partial credit.
- Budget exhaustion differs from semantic acceptance. In the paper's bounded
  reply-revision path, proceeding to completeness checks after the allowed
  revision is not evidence of a new semantic PASS.

## Validation and Delivery

Reuse existing tests and choose checks that distinguish correct behavior from
the relevant failure modes.

| Change | Validation focus |
|---|---|
| State machine or rework | Normal path, malformed output, unparseable decisions, exhausted budgets, actual return stage |
| Sampling or resume | Reproducible seeds, excluded combinations, preserved assignments, no duplicate exports |
| Multi-turn orchestration | Final-turn branch, earlier-media references, dependency invalidation, retained valid turns |
| Templates or validation | Loadable configuration, real rendering contexts, missing variables, conflicting rules |
| Replies or rubrics | One Tier 0, tier prerequisites, partial credit, unsupported factual claims |
| Media backend | Readable synchronized audio and video, duration, failure propagation |

An existing `--test-run` may still call a text backend; it is not synonymous with
offline testing. For framework-only work, report what offline checks establish
and what remains unverified. When live integration and a budget are already
authorized, complete the corresponding small-scale checks.

Update affected interface documentation, configuration examples, and module
mappings. Follow the target repository's review and commit conventions. Report
the changes, observed behavior, actual validation, and remaining limitations
without presenting offline success as proof of synthesis quality.
