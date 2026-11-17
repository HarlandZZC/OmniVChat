# Extending the Multi-turn Subsystem

## Modules and Files

The paper's path begins with Attribute & Corpus Sampling, Description, Plan,
and Plan Review. The Per-turn Agent Loop then runs Prompt Generation, Prompt
Review, Video Generation, Video Caption, Video Review, and conditional Video Q-A.

Intermediate turns run Reply Generation and Reply Review before Go to Next
Turn. The final turn runs Final-Reply Plan before Reply Generation and Reply
Review. After assembly, the full dialogue undergoes Validation and conditional
Validation Review. Update Plan revises the global plan and redoes affected turns.

A compatible implementation commonly uses:

```text
multi-round-unified/subcategories/<subcategory-id>/
  spec.yaml
  extra_requirements.yaml
  prompts/
    system_prompt.md.j2
    describe.md.j2
    describe_fallback.md.j2
    plan.md.j2
    generate.md.j2
```

The public module name is **Prompt Generation**, even if its template is named
`generate.md.j2`. Plan Review, Prompt Review, video, and reply templates are
usually shared framework resources.

## Define the Historical Dependency

Plan the new facts, user actions, reply directions, and media dependencies in
each turn. Specify:

- Which earlier turn establishes an object, speaker, scene, identity, or state.
- Which intermediate facts change and which must remain stable.
- What the final turn retrieves, updates, compares, or preserves.
- Why the target cannot reliably be completed without history.
- Which facts historical replies may contain, and whether they would turn an
  audio-visual task into text-only lookup.

A historical dependency need not be an abrupt reversal. Identity consistency,
state updates, and delayed retrieval can use different structures. Earlier
turns should have natural user behavior and reasons for replying.

For visual retrieval tasks, avoid having historical replies enumerate every
target detail. Apply this constraint to the specific capability rather than
prohibiting all reasonable references to scene facts.

## Template Variables

These are compatibility hints for the existing layout. Confirm actual call
sites in the implementation being edited.

| File | Available variables |
|---|---|
| `system_prompt.md.j2` | `dimensions_section`, `max_duration` |
| `describe.md.j2` | `attr_block`, `corpus_samples`, `corpus_count`, `corpus_safety_rules`, `extra_requirements`, `eval_category`, `dimensions_section`, `total_rounds`, `max_duration` |
| `describe_fallback.md.j2` | `attr_block`, `extra_requirements`, `eval_category`, `dimensions_section`, `total_rounds`, `max_duration` |
| `plan.md.j2` | `description`, `inspiration`, `attr_block`, `attributes`, `extra_requirements`, `eval_category`, `dimensions_section`, `total_rounds`, `max_ref_chain_depth`, `max_duration`, `experience_block` |
| `generate.md.j2` | `round_idx`, `total_rounds`, `is_last_round`, `round_plan`, `presets_desc`, `preset_keys`, `materials_catalog`, `conversation_history`, `attributes`, `eval_category`, `generate_retry_context`, `default_resolution`, `max_duration`, `validation_rules_block` |

Plan may receive `plan_retry_context` only on revisions or parse retries. Do not
assume that `experience_block` contains additional knowledge; it may be empty.

The system prompt has no `total_rounds`; Description has no `attributes`
dictionary; Prompt Generation has neither `attr_block` nor `dimensions_section`
in this contract. Shared conversation memory does not automatically create
Jinja variables.

## Plan and Plan Review

Plan emits a structured plan for all turns. Follow the base subcategory and
parser contracts for global characters and each turn's `round`, content, scene,
`reply_sketch`, and asset strategy. Use supported duration and environment-change
structures.

The `rounds` array must match the sampled turn count. Give each character a
stable identity. For speaker, camera, or environment changes, state what changes
and what remains. Target facts must be confirmable in the eventual media, not
merely written into hidden plan fields.

Plan Review checks the whole sequence and its references. Put capability-specific
conditions in the system prompt and Plan's self-check instructions so the shared
review can assess them. Override shared review templates only when requirements
justify doing so.

## Prompt Generation and Media References

Use `is_last_round` to distinguish the current turn's purpose and generate its
content from `round_plan`, `materials_catalog`, and `conversation_history`.
Preserve the active schema for `user_turn` and `video_content`. A compatible
implementation expects exactly one nonempty text entry, plus supported media
references.

Reference placeholders can include:

```text
{{round_1_video_url}}
{{round_1_audio_url}}
{{round_1_last_frame_url}}
```

The number identifies an actual earlier turn. These are placeholders in model
output, not automatic Jinja variables. Display them in templates using the
existing escaping convention or:

```jinja
{% raw %}{{round_1_video_url}}{% endraw %}
```

References resolve at execution time. Turn 1 has no earlier media; later turns
need valid earlier references. Reject future, unfinished, or invalidated assets.
Check the resolved references actually submitted to the backend.

Consider reference selection along four axes:

- **Visual facts:** does an earlier clear asset still contain the required
  objects and scene?
- **Action continuity:** does a changed scene or ongoing action require newer
  evidence?
- **Appearance and voice:** when using a last frame for appearance, provide the
  corresponding character's audio as needed.
- **Executability:** are duration, count, and media combinations supported by the
  current Renderer?

Fixed reference limits and rendering capabilities are backend-specific. Prompt
Review checks references before rendering; Video Review checks the resulting
media against the plan.

## Turn Counts, Language, and Field Rules

Sampled `diversity.rounds` is the primary turn-count source in the compatible
runner, with `default_rounds` as fallback. Values must agree with
`multi_round.min_rounds`, `max_rounds`, and templates. Metadata limits do not
necessarily clamp invalid samples. Bind sequence lengths to turn counts with
feasibility rules when needed.

The multi-turn assembler may fill some `scenario.*` fields from sampled
attributes and use a global character registry. Preserve these mappings instead
of copying single-turn `global_scene_setup.characters[0]` paths.

Typical synthesis field rules:

```yaml
validation:
  per_round_rules:
    default:
      required_fields: [ai_response]
    last:
      required_fields: [ai_response, key_points]
```

Synthesis produces a Reference Reply for every turn and a Rubric for the last.
Benchmark export includes intermediate replies as `reference_reply` and keeps
the final rubric in `key_points`, while excluding the final reference reply.
`language` controls the final reply and Tier 0; multilingual history needs a
separate attribute.

`required_round_dialogue_keywords` can detect dialogue cues in specified turns.
It cannot establish the correct retrieval target or speaker identity. Separate
mechanical absence checks from semantic judgments requiring actual media.

## Update Plan and Rework

Use current-turn Prompt Generation for local text defects and Re-seed for
random rendering failures. If observed media invalidates the global plan,
Update Plan preserves sampled language, turn count, and capability attributes,
versions the plan, and restarts at the earliest affected turn.

Check that obsolete references are invalidated, dependent later results are not
reused, valid earlier turns remain, and final conversation length is correct.
Instructions alone do not establish these guarantees; framework changes need
behavioral validation.

## Semantic Acceptance and Evaluation

Static checks should cover the first, intermediate, and final turns, plus
retries. Compare a case with the necessary history against one where the decisive
historical fact is removed or changed.

Perform real integration within the authorized scope. Inspect objects,
characters, voices, scenes, and orientation turn by turn, then check the final
capability, reply language, and rubric.

Construct benchmark context as user clip 1, its fixed reference reply, user clip
2, its fixed reference reply, and so on, ending with the final user clip.
Include each clip's audio. Generate and score only the final model response;
the final reference reply and rubric are not model inputs. If real media has
not been inspected, report only the configuration and workflow checks actually
completed.
