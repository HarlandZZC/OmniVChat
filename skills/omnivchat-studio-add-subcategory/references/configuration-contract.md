# Subcategory Configuration and Validation

This reference covers both subsystems. Relevant entry points in a compatible
implementation include `core/config.py`, `core/diversity.py`, `core/prompts.py`,
`core/validator.py`, `core/validator_rules.py`, and `core/agent.py`. Verify actual
field readers and call behavior in the target repository.

## Capabilities and Display Names

The paper's five Ability Categories are:

| Name | Code | Distinction |
|---|---|---|
| Dialogue-State & Link Perception | DSLP | Connection, turn boundaries, camera orientation, and dialogue state |
| Multimodal Entity Alignment | MEA | Aligning speech, references, and interaction roles with the correct entities |
| Model Self-Awareness | MSA | Model identity and physical capability boundaries |
| Anti-Hallucination | AH | Grounding replies in available audio-visual evidence |
| Emotion Recognition | ER | Recognizing emotion and responding appropriately |

This is the paper's terminology, not a closed registry forbidding new
capabilities. State the new subcategory's parent; justify a new Ability Category
when needed.

Historical IDs may retain `ISLP`, `MICC`, `VIMC`, or `MEU-ER`. Their corresponding
display terms are `DSLP`, `MDCC`, `VDMC`, and `ER`. Apply explicit mappings instead
of unrestricted replacement in historical files and records.

A Scenario Domain describes what a conversation discusses; a Subcategory
describes what it tests. Use stable lowercase-hyphen directory IDs and keep
`eval_category` compatible with the loader and displays.

## Specification

Typical `spec.yaml` fields:

| Field | Purpose and checks |
|---|---|
| `eval_category` | Stable capability ID consistent with templates, exports, and taxonomy |
| `input_modality` | An input modality accepted by the actual implementation |
| `diversity` | Dimensions, values, descriptions, distributions, and output mappings |
| `diversity.preferences` | Combination weights or feasible-set restrictions |
| `validation` | Duration, speech rate, ending constraints, and `rules` |
| `multi_round` | Multi-turn settings consistent with sampled `rounds` |
| `validation.per_round_rules` | Intermediate-turn and final-turn field requirements |
| `agent_overrides` | Necessary control overrides supported by the runner |

A compatible loader normalizes `diversity` into `dimensions` and flattens
`validation` into specification fields. A custom YAML field may parse without
having any consumer; identify the reader before adding one.

Common dimensions include `language`, `num_people`, `scene_category`,
`camera_setup_base`, `audio_interference`, `visual_interference`, and `ethnicity`;
multi-turn configurations also include `rounds`. Follow the actual base
subcategory rather than requiring a fixed dimension count. Every template
variable needs an explicit source.

A dimension usually has `values` and `distribution`. Values may carry
`description` and `warnings`; use `json_path` when there is a reliable output
location. The following illustrates two custom dimensions, not a complete
subsystem specification:

```yaml
diversity:
  target_position:
    distribution: uniform
    values:
      left:
        description: The target is to the user's left.
      right:
        description: The target is to the user's right.
  cue_form:
    distribution: uniform
    values:
      gesture:
        description: The user indicates the target with a visible gesture.
      description:
        description: The user identifies the target by describing its appearance.
  preferences:
    - match:
        target_position: left
        cue_form: gesture
      weight: 1.5
```

This is a probability preference example, not a claim that left-plus-gesture is
always desirable. Justify weights from the design rather than selecting them to
obtain preferred outcomes.

## Fixed Conditions, Distributions, and Combinations

- Keep core capability prerequisites fixed. For an absent-object task, actual
  absence from the visible scene cannot depend on a rare optional requirement.
- A new dimension should change evidence, ambiguity, temporal dependency, or
  state judgment, rather than list synonymous settings.
- Keep numeric key types compatible with templates and sampling. Turn-count
  values must convert to valid integers.
- Calculate the Cartesian-product size before enumerating combinations.
- `weight: 0` excludes matching combinations. Ordinary positive preference
  multipliers multiply. `exclusive: true` forms allowed-combination groups:
  OR within a group, AND across groups. Incompatible groups yield an empty pool.
- If demand exceeds the positive-weight pool, a compatible sampler may repeat
  selection with fresh keys. Do not promise uniqueness across repeated batches.

`json_path` points into generated output. `compute: speech_segments` supports
single-turn dialogue-segment counting where implemented; invented `compute`
values do not take effect automatically. Multi-turn assembly may fill some
`scenario.*` fields from sampled attributes. Matching metadata does not prove the
rendered media realizes those attributes.

Distinguish target configuration, constrained sampling, completed data, and
human-confirmed semantics. Failures and filtering may alter the final
distribution.

## Templates and Rules

Prefer subcategory templates, falling back to `core/templates/`. Shared video
review and reply templates usually need no copying. If overridden, preserve
their field and decision protocols. System-prompt contexts differ from ordinary
stage contexts; consult the subsystem references.

Use the Validator for structure, enums, required fields, continuous timing, and
mechanically checkable cues. Use Director review, supported by Reviewer evidence,
for actual visibility, unique references, necessary history, camera-relative
left/right, and natural behavior.

Reflect rules in both generation instructions and output checks. A keyword in
the user's speech does not prove that the target capability is present.
Vocabulary checks should cover allowed languages and valid expressions, with
false positives tested. Verify the current meaning of `position` in rules such
as `required_ending_cues` and `required_round_dialogue_keywords`, especially the
distinction between a timeline segment and a turn.

Rendering constraints are backend-specific. Inherit the runner's actual duration,
reference-count, and media-combination constraints. Choose evidence that it can
present clearly.

## Extra Requirements

`extra_requirements.yaml` supplies optional variations using the reader's
supported list or grouped-list format; an empty list is valid. Each entry should
add one compatible variation without replacing language, people count, turn
count, or core truth, and without expanding the story beyond the clip duration.
Persist whether an extra requirement was sampled and which one.

## Shared Rubric Conventions

Use a `key_points` list with `tier` and `point` in each criterion. Exactly one
Tier 0 criterion specifies reply language. It earns no credit, and failure
zeroes the rubric score. Later tiers contribute only when all earlier tiers are
fully satisfied; eligible criteria in the same tier earn credit individually.

Each criterion must have a clear truth value supported by the current clip or
valid history. The Reference Reply should satisfy all criteria, without requiring
other correct answers to repeat its wording. Hidden sampling labels, script
intent, or review notes must not become required evidence for the tested model.

For multi-turn tasks, `language` controls the final reply's language. Use a
separate attribute for different historical languages instead of silently
changing the meaning of `language`.

## Offline Checks

Run within the relevant subsystem. Replace `new-subcategory` with the actual
directory. This example reads the subcategory specification without creating
generation backends or loading global runtime configuration:

```python
from math import prod
from pathlib import Path
from core.config import SubcategorySpec
from core.diversity import build_dimensions, generate_combinations

subcategory = "new-subcategory"
spec = SubcategorySpec.from_yaml(
    Path("subcategories") / subcategory, subcategory
)
dims, values = build_dimensions(spec)
assert dims and all(values[name] for name in dims)
space = prod(len(values[name]) for name in dims)
print("candidate_assignment_count", space)
# Check that the enumeration size is appropriate before sampling.
targets = {name: spec.get_dim_distribution(name) for name in dims}
samples = generate_combinations(
    12, dims, values, targets=targets,
    preferences=spec.diversity_preferences, seed=7,
)
assert len(samples) == 12
assert all(set(sample) == set(dims) for sample in samples)
print("sampled_assignments", samples)
```

This establishes that specification-level sampling works. Batch jobs may impose
additional coverage targets or overrides. Check valid values and exclusions for
new dimensions, and verify feasible sets with small configurations.

For each new template:

1. Parse its Jinja syntax.
2. Identify available variables from the actual `render()` call in `core/agent.py`.
3. Render realistic offline contexts for initial attempts, retries, and relevant
   intermediate-turn and final-turn branches.
4. Use strict undefined handling for required variables; supply empty values or
   `is defined` guards for intentionally optional ones.
5. Inspect the rendered meaning, not merely the absence of an exception.

Some inherited templates rely on conversation memory; a Description result may
not be passed again as a separate variable. Do not invent a variable that its
caller does not supply.

`python cli.py ... --dry-run` may load runtime configuration before previewing
assignments. `--test-run` may still access a text backend. Real video and
semantic quality require separately authorized integration checks.
