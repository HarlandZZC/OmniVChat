# Extending the Single-turn Subsystem

## Modules and Files

Implement the new capability primarily in Description, Plan & Review, Script
Generation, and Script Review. The remaining path uses Validation, conditional
Validation Review, Video Plan, Video Generation, Video Caption, Video Review,
and conditional Video Q-A. Accepted media proceeds to Reply Plan, Reply
Generation, and Reply Review. These names follow Figure 2 of the paper.

A compatible implementation commonly uses:

```text
single-round-unified/subcategories/<subcategory-id>/
  spec.yaml
  extra_requirements.yaml
  prompts/
    system_prompt.md.j2
    describe.md.j2
    describe_fallback.md.j2
    plan_and_review.md.j2
    generate.md.j2
    script_review.md.j2
```

Select files from an actual structurally similar subcategory and rewrite all
capability-specific content. The loader supplies shared video and reply
templates; a new subcategory does not require copying them.

## Template Responsibilities and Inputs

Confirm these compatibility hints against current call sites.

| File | Responsibility | Available variables |
|---|---|---|
| `system_prompt.md.j2` | Director context, capability, and global requirements | `dimensions_section`, `max_duration` |
| `describe.md.j2` | Description: design an attribute-compatible scene from corpus | `attr_block`, `corpus_samples`, `corpus_count`, `corpus_safety_rules`, `extra_requirements`, `eval_category`, `dimensions_section`, `max_duration` |
| `describe_fallback.md.j2` | Description without matching corpus | `attr_block`, `extra_requirements`, `eval_category`, `dimensions_section`, `max_duration` |
| `plan_and_review.md.j2` | Plan & Review: refine and self-check the scenario | `eval_category`, `extra_requirements`, `attributes`, `max_duration` |
| `generate.md.j2` | Script Generation: structured scene and timeline | `eval_category`, `attributes`, `num_people`, `generate_retry_context`, `validation_rules_block`, `max_duration` |
| `script_review.md.j2` | Script Review: determine whether the capability is tested | `json_str`, `eval_category`, `extra_requirements`, `num_people`, `script_review_retry_hint` |

Description receives formatted `attr_block`, not an `attributes` dictionary.
Script Review has no general `attributes` variable; derive facts from the
candidate `json_str` and existing context. Matching names do not automatically
bind a dimension, output field, and template variable.

## Stage-specific Content

**Director system prompt.** Define the capability, correct behavior,
characteristic errors, neighboring boundaries, required evidence, and acceptable
reply scope. Preserve current formatting, role, timing, and media constraints.
Use examples to explain concrete difficulties.

**Description.** Develop a natural user motive and renderable scene from allowed
corpus. Follow the parser's `MATCH` / `NO_MATCH` protocol. A fallback must meet
the same capability and attribute requirements. Corpus provides inspiration,
not hidden evidence for the tested model.

**Plan & Review.** Identify decisive objects, actual properties, the user's
claims, and expected behavior. Check whether the clip can establish truth,
whether references are unique, whether the question is answerable from the
available input, and whether the user yields the turn appropriately. A
wait-for-the-user task should not inherit an incompatible requirement that the
user has finished a complete question.

**Script Generation.** Emit the active schema, typically including:

| Field | Content |
|---|---|
| `scenario` | Scene description, people count, capability ID, Scenario Domain |
| `global_scene_setup` | Camera view, environment, lighting, characters, interference |
| `unified_timeline` | Continuous, nonoverlapping `time_range` entries and `content` |

Preserve the script's delimiter protocol: `{}` for speech, `<>` for sound
effects, fullwidth parentheses (U+FF08/U+FF09) for music, and corner brackets
(U+3010/U+3011) for subtitles. A schema's subtitle support does not authorize
putting answers on screen. Follow the task's actual configuration and templates.
Align character, device, camera, speech-rate, and ending requirements with the
Validator.

Keep `generate_retry_context` and `validation_rules_block` in real templates.
If the base uses `**...**` to mark important facts for human readers, keep the
marks in scene descriptions, outside spoken dialogue and tested-model input.

**Script Review.** Follow the parser's `PASS`, `REWRITE`, and `GIVE_UP` protocol.
Assess target truth, neighboring boundaries, shortcuts, renderability, and
whether rework is justified. Return a decision that changes the next action.

## Align Dimensions with Ground Truth

Check the base subcategory's `json_path` mappings. Single-turn language and
character properties commonly live under `global_scene_setup.characters[0]`;
camera and interference fields live under `global_scene_setup`. Do not copy
multi-turn `scenario.*` mappings blindly.

Where implemented, `timeline_segments` can use `compute: speech_segments` to
count dialogue segments, not every timeline interval. Compare speech rate,
duration, and ending tolerances across configuration and templates instead of
copying conflicting numbers from another subcategory.

Examples of capability-specific evidence checks:

- **Absent object:** the queried object is outside the visible scene, and the
  reply's refusal follows from that evidence.
- **False premise:** the user's assertion conflicts with visible facts, rather
  than merely asking an open question.
- **Left/right:** define user-relative versus frame-relative directions and
  check the mapping after orientation changes.
- **Reference or speaker resolution:** plausible candidates exist and cues
  distinguish them; a sole candidate should not make guessing trivial.
- **Emotion:** actual expression or voice supports the label; writing an emotion
  name into a plan does not establish successful rendering.

These are design examples, not simultaneous requirements for every subcategory.

## Replies and Acceptance

Shared Reply Generation writes `ai_response` and `key_points`. Script Generation
should not preempt it with final annotations that have not been checked against
the media. Reply Plan and Reply Review use accepted captions, Q-A, and context,
without treating missing script details as observed facts.

After static checks, validate concrete samples within the authorized scope:
the decisive facts are present in audio and video, the interaction is natural,
the capability is distinguishable, and both Reference Reply and Rubric follow
the actual media. Inspect failure and rework reasons, then fix demonstrated
problems without weakening the capability to increase completion counts.
