# Compatibility Map for Existing Implementations

Locate `single-round-unified/` and `multi-round-unified/` in the repository
specified by the user. They correspond to the Single-turn Subsystem and
Multi-turn Subsystem. The paths below are compatibility hints for that layout,
not dependencies shipped with this skill. Confirm actual files and call sites
before using them.

## Module Entry Points

Both subsystems commonly orchestrate work in `core/agent.py`; multi-turn work also
uses `core/state.py`. Method names help locate code and do not replace public
module names.

| Paper module | Existing implementation location |
|---|---|
| Attribute & Corpus Sampling | `core/diversity.py`, `core/corpus.py`, `cli.py`, `core/batch.py` |
| Description | `_do_describe()`, `_do_describe_fallback()` in `core/agent.py` |
| Plan & Review | Single-turn `_do_plan_and_review()` |
| Plan / Plan Review | Multi-turn `_do_plan()`, `_do_plan_review()` |
| Script Generation / Prompt Generation | Each subsystem's `_do_generate()` |
| Validation | `_do_validate()`, `core/validator.py`, `core/validator_rules.py` |
| Validation Review | `_do_validate_review()` |
| Script Review | Single-turn `_do_script_review()` |
| Prompt Review | Multi-turn `_do_prompt_review()` |
| Video Plan | Single-turn `_do_video_plan()` |
| Video Generation | `_do_video_generate()`, `core/video_client.py` |
| Video Caption | `_do_video_caption()`, `core/reviewer_client.py` |
| Video Q-A | The question branch in Video Review; multi-turn may use `_handle_reviewer_question()` |
| Video Review | `_do_video_review()`, `_handle_video_review_response()` |
| Reply Plan / Final-Reply Plan | Each subsystem's `_do_final_reply_plan_and_review()` |
| Reply Generation / Reply Review | `_do_final_reply_generate()`, `_do_final_reply_review()`; intermediate multi-turn replies use `_do_mid_reply_*()` |
| Go to Next Turn | Multi-turn `_do_round_advance()` |
| Update Plan | Multi-turn `_handle_update_plan()`, `_restart_from_round()` |

Other relevant locations:

- `core/config.py`: `SubcategorySpec` reads subcategory specifications;
  `AgentConfig` describes runtime controls.
- `core/prompts.py`: `TemplateLoader` prefers subcategory templates and falls back
  to `core/templates/`.
- `core/system_prompt_builder.py`: shared dimension descriptions.
- `core/batch.py`: batch tasks, checkpoints, completion, and failure.
- `core/media_tools.py`: media checks, derived audio, and last frames.

## Templates and Configuration

Subcategories live under `subcategories/<subcategory-id>/`. Single-turn
subcategories typically provide `system_prompt`, `describe`, `describe_fallback`,
`plan_and_review`, `generate`, and `script_review`. Multi-turn subcategories
typically provide `system_prompt`, `describe`, `describe_fallback`, `plan`, and
`generate`. Files use `.md.j2`; shared video and reply templates remain in each
subsystem's `core/templates/`.

A compatible loader normalizes `spec.yaml`'s `diversity` into `dimensions`,
handles `preferences` separately, and flattens `validation` limits and rules into
the specification. Trace fields to their consumers; valid YAML alone does not
prove a setting takes effect.

Legacy Jinja rendering may tolerate undefined variables. Validate real keys,
nested values, and branches. Explicitly account for optional variables, such as
an absent `plan_retry_context` on the initial Plan call.

## Existing Result Fields

| Content | Single-turn | Multi-turn |
|---|---|---|
| Scene | `scenario`, `global_scene_setup` | `scenario`, `characters`, `scenes` |
| User content | `unified_timeline` | `conversation[i].user_turn` |
| Accepted video | Top-level `video` | `conversation[i].video` |
| Reference Reply | Top-level `ai_response` | Every `conversation[i].ai_response` |
| Rubric | Top-level `key_points` | Final `conversation[-1].key_points` |
| Criterion | `tier` and `point` in each `key_points` entry | Same |

`scenario.total_rounds` must match the conversation length. Synthesis uses
completed per-turn replies when planning later turns. Benchmark export retains
the intermediate replies in `turns[i].reference_reply`, and model input excludes
the final reference reply and rubric. Keep `key_points` compatible with its
consumers when changing display terminology.

## Behavior to Verify

- `--dry-run` may preview assignments only after loading global runtime
  configuration. Fully offline checks can call `SubcategorySpec.from_yaml()` and
  sampling utilities directly.
- `--test-run` may mock video while still using a real text backend. It does not
  establish semantic quality or real-media validity.
- Turn count may come from sampled `rounds`, falling back to `default_rounds`.
  Confirm where `min_rounds` and `max_rounds` are enforced.
- Multi-turn Prompt Generation may run shape checks before Prompt Review.
  Invalid content should return to the appropriate generation stage.
- Later turns need valid resolved media references. A placeholder in the prompt
  does not prove the backend received a usable reference.
- Bounded Reply Review and final completeness checks do not mean that every
  semantic review passed.
- Read retry budgets and media limits from current configuration and supported
  overrides, rather than treating old examples as universal settings.

## Maintenance Scope

When state or schemas change, inspect batch handling, templates, validators,
export consumers, and resume logic. Update corresponding configuration examples
when their structure changes, while keeping runtime values private.

Display mappings should preserve historical data. Routine subcategory additions
usually need specifications and templates. If the framework lacks a capability,
add a reusable mechanism rather than branching on one subcategory's name.
