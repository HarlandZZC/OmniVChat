# OmniVChat-Studio Modules and Data Contracts

Use the names from Figure 2, Sections 2.1-2.2, and Appendix C of the
[OmniVChat paper](https://arxiv.org/abs/2609.21465). Code may use corresponding
snake_case identifiers; historical abbreviations should not replace public names.

## Four Roles

| Role | Input | Output and responsibility |
|---|---|---|
| Director | Subcategory requirements, sampled attributes, history, plans, evidence | Scenario design, planning, text generation, semantic review, rework decisions |
| Validator | Structured candidates and configuration rules | Deterministic pass or violations with field locations and reasons |
| Renderer | Rendering prompt, valid earlier-media references, optional seed | Synchronized audio and video, inspectable synthesis status |
| Reviewer | Actual audio and video, optional focused questions | Caption, quality report, media-grounded Q-A evidence |

The Reviewer does not decide acceptance. The Director should cite observations
and preserve uncertainty rather than treating a perception error as a confirmed
rendering defect.

## Single-turn Subsystem

| Module | Owner | Contract and normal exit |
|---|---|---|
| Attribute & Corpus Sampling | Deterministic utilities | Fix an attribute assignment and corpus candidates, then enter Description |
| Description | Director | Design a scenario from attributes and corpus; bounded resampling or an explicit fallback if no corpus matches |
| Plan & Review | Director | Refine and check the scenario within one planning process, then enter Script Generation |
| Script Generation | Director | Produce a structured scene and continuous timeline, then enter Validation |
| Validation | Validator | Enter Script Review on success, or Validation Review on violations |
| Validation Review | Director | Enter Script Review if all reported issues are semantic false positives; otherwise return reasons to Script Generation |
| Script Review | Director | Check the capability, plausibility, and answerability; accept, rewrite, or give up |
| Video Plan | Director | Combine the script with compatible camera, continuity, and realism requirements |
| Video Generation | Renderer | Produce synchronized media; send success to Video Caption and failure to Video Review |
| Video Caption | Reviewer | Describe actual media and its quality, then enter Video Review |
| Video Q-A | Reviewer | Answer focused questions from Video Review and return evidence there |
| Video Review | Director | Compare evidence with the target; accept, Re-seed, Re-write, ask questions, or give up |
| Reply Plan | Director | Plan a reply from accepted media evidence, excluding unconfirmed script details |
| Reply Generation | Director | Produce a Reference Reply and instance-level Rubric |
| Reply Review | Director | Check language, evidence, tier ordering, and naturalness; accept or revise within a budget |

Video Q-A is conditional. Validation Review runs only when Validation reports
issues.

## Multi-turn Subsystem

The shared opening is:

```text
Attribute & Corpus Sampling -> Description -> Plan -> Plan Review
```

Plan describes all turns, characters, scenes, reply directions, and media
dependencies. Plan Review independently checks the whole plan before entering
the Per-turn Agent Loop.

```text
Prompt Generation -> Prompt Review -> Video Generation -> Video Caption -> Video Review
```

Video Review may request Video Q-A, Re-seed, a current-turn Re-write, or Update
Plan. After acceptance:

```text
Intermediate turn: Reply Generation -> Reply Review -> Go to Next Turn
Final turn:        Final-Reply Plan -> Reply Generation -> Reply Review -> assembly
```

The assembled dialogue proceeds through:

```text
Validation -> conditional Validation Review -> completeness checks
```

| Multi-turn module | Contract |
|---|---|
| Plan | A global structured plan with consistent turns, characters, and references |
| Plan Review | Independently accept, revise, or abandon the global plan |
| Prompt Generation | Build rendering input from the current plan, history, and asset catalog; incorporates the single-turn Script Generation and Video Plan responsibilities |
| Prompt Review | Check content, references, and the current turn's objective before rendering |
| Final-Reply Plan | Run only on the final turn, using full history to plan the final reply and rubric |
| Go to Next Turn | Advance after the current turn passes; control flow, not another agent role |
| Update Plan | Version the revised global plan and restart from the earliest affected turn |

The remaining module names are shared with the single-turn subsystem, with
current-turn plans, completed history, or the whole dialogue added to their
inputs. Assembly and completeness checks are implementation duties, not extra
paper-defined synthesis agents.

## Data Objects

These are conceptual contracts. See [legacy-map.md](legacy-map.md) for
compatibility with an existing implementation.

| Object | Required information |
|---|---|
| Subcategory Configuration | Capability, stable ID, dimensions, distributions, preferences, templates, rules, applicable turn limits |
| Sample Assignment | Stable task ID, full sampled attributes, randomness, provenance of corpus and extra requirements |
| Plan | Scene and target facts; global characters, per-turn content, reply directions, and asset dependencies |
| Script or Prompt | User audio-visual content, timing, character mapping, valid earlier-media references |
| Render Result | Success or failure, usable media identifier, actual duration, audio status, attempt metadata |
| Evidence | Caption, quality observations, and Q-A for accepted media, retaining uncertainty and contradictions |
| Reference Reply | Text grounded in current media and valid history, in the required language |
| Rubric | Individually decidable criteria with tiers; the reference should satisfy all criteria |
| Run State | Stage, accumulated reasons, budgets, plan version, completed turns, dependencies, completion status |

Export public dialogue records through an explicit field allowlist. Runtime
state and service metadata should not flow into the dataset automatically.
For benchmark export, retain intermediate Reference Replies as
`turns[i].reference_reply`, omit the final reference answer, and keep the final
Rubric in top-level `key_points` for scoring only.

## Sampling and Coverage

For an attribute assignment `a`, combine marginal dimension weights and matching
preference multipliers:

```text
w(a) = product_d p_d(a_d) * product_matching_preferences m
k(a) = log(U) / w(a), U in (0, 1)
```

Excluded combinations have zero weight. Select the largest `n` keys among
positive-weight combinations for weighted sampling without replacement.
If more assignments are requested than feasible combinations, repeat selection
with fresh random keys; combinations may repeat across batches. Handle an empty
feasible set explicitly.

In a compatible configuration, `exclusive` entries are grouped by the set of
matched dimensions. Entries in one group form an OR; groups combine with AND.
Ordinary preference multipliers multiply. Verify these semantics in the target
sampler instead of treating `exclusive` as a probability boost.

Compute the Cartesian-product size before adding dimensions. Extend an existing
dimension when a proposed new one is strongly correlated. Report intended
marginals, constrained assignment distributions, and completed-sample
distributions separately.

## Rework and Resume

| Trigger | Return point | Retained and invalidated state |
|---|---|---|
| Structure, timing, or configuration error | Script Generation, or the affected multi-turn stage | Preserve attributes and a valid global plan; accumulate reasons |
| Script or rendering-prompt defect | Script Generation or current-turn Prompt Generation | Invalidate reasoning, media, and evidence derived from the discarded draft |
| Random rendering defect | Video Generation with a fresh seed | Preserve the accepted script; replace the rendering and observations |
| Insufficient media evidence | Video Q-A, then Video Review | Preserve media and add focused evidence |
| Invalid multi-turn global plan | Update Plan | Preserve sampled attributes; invalidate and redo from the earliest changed turn |
| Reply defect | Reply Generation | Revise within a budget without regenerating accepted video |

A rewrite may reset local rendering or question counters, while a global budget
still bounds total work. Record each transition, its evidence, and valid outputs.

Single-turn rewrites should not retain abandoned script reasoning. When redoing
turn N, preserve valid turns before N and invalidate turn N and dependent later
results. Resume checks must consider configuration version, task identity, and
asset validity, rather than trusting a matching filename.

## Media and Evidence

Renderer interfaces specify what to render and what to reference; operators
provide the backend. A completed turn can yield audio and a last frame for later
reference. Validate ordering, existence, duration, and compatible media types.
Unresolved references must not silently become empty evidence.

An earlier clear asset may remain useful in a static scene. Scene changes and
ongoing actions require evidence of the new state. A still image cannot carry a
voice; preserve character voice with appropriate audio references. These are
planning criteria, not guarantees of rendering fidelity.

Video Caption observes cuts, realism, speech rate, body structure, appearance
consistency, visual artifacts, and on-screen text. Video Review determines
whether a discrepancy affects the capability. Incidental background changes may
be acceptable; facts that determine the answer require resolution.

## Rubric

Use exactly one Tier 0 criterion for reply language. It earns no credit, and
failure makes the rubric score zero. Later criteria must each carry one specific
judgment grounded in allowed evidence. Put core correctness before additional
detail or scenario-specific expression.

A tier contributes only when all earlier tiers are fully satisfied. Within an
eligible tier, satisfied criteria earn credit individually. If one of two Tier 2
criteria passes, it can earn credit while Tier 3 remains blocked. Synthesis-time
Reply Review and benchmark scoring are separate operations.

## Completion Status

Record structural completeness, media status, and semantic review separately.
The paper's bounded reply revision may proceed to final completeness checks
after the allowed revision; this does not establish a later semantic PASS.
Distinguish offline tests, text-backend integration, real-media integration, and
human semantic inspection in reports.
