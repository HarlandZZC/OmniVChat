---
name: omnivchat-studio-add-subcategory
description: Add or extend single-turn and multi-turn OmniVChat-Studio subcategories, including capability definitions, configurations, diversity dimensions, prompt templates, validation rules, and targeted checks. Use for new evaluation capabilities, interaction subcategories, or history-dependent extensions of single-turn tasks.
---

# OmniVChat-Studio: Add a Subcategory

Turn a clearly defined interaction capability into a Subcategory Configuration
that can be synthesized, distinguished from neighboring capabilities, and
validated within an existing OmniVChat-Studio subsystem.

## Terminology and Reading Order

Follow Figure 2, Section 2, and Appendix C of the
[OmniVChat paper](https://arxiv.org/abs/2609.21465). Use **OmniVChat-Studio** and
the roles **Director, Validator, Renderer, Reviewer**. Distinguish **Ability
Category, Subcategory, Scenario Domain, Reference Reply, Rubric, Tier, Criterion**.

1. Read the target repository's instructions, taxonomy, subsystem interfaces,
   and output schemas.
2. Read [configuration-contract.md](references/configuration-contract.md).
3. Read the relevant subsystem reference:
   [single-turn.md](references/single-turn.md) or
   [multi-turn.md](references/multi-turn.md).
4. Inspect the actual specification and all templates of the closest structural
   example. Compare a neighboring subcategory that could be confused with it.
   Documentation summaries do not replace templates and call sites.

Distribute capability design, configuration rules, templates, and self-contained
examples. Keep credentials, runtime configuration, raw service logs, deployment
addresses, personal workspace information, and unpublished provider details out
of subcategories and skill packages.

## Define a Distinct Capability

Extract the following into a short design note. Reuse information already given;
clarify missing inputs only when they change the capability being designed.

| Design item | Question to resolve |
|---|---|
| ID and name | What is the name, Ability Category, and stable directory ID? |
| Target behavior | What cue does the user provide, what should the model do, and what is the characteristic error? |
| Evidence and truth | Which audio, visual, or historical fact determines the answer, and how will actual media confirm it? |
| Modalities and history | Which modalities are necessary, and why does a multi-turn task require history? |
| Evaluation point | Is the target a single turn or the final turn, and when must each fact appear? |
| Neighboring boundary | What new judgment distinguishes it from the closest existing subcategory? |
| Scenario variation | Which variations change the reasoning problem, and which merely change people or scenery? |

A different Scenario Domain alone does not establish a new capability.
Multi-turn tasks need not follow a fixed final-turn reversal story; the last
turn should exercise the requested retrieval, state update, identity
consistency, or other explicit capability.

## Implementation Workflow

1. **Choose the subsystem.** Single-turn judgment uses the current clip;
   multi-turn judgment explicitly depends on earlier evidence or state. Preserve
   existing directories and historical identifiers, using mappings for display
   names.
2. **Design scenarios and variables.** Start with substantively different
   scenarios, then identify a few independent dimensions. Separate fixed
   capability prerequisites, sampled attributes, optional requirements, and
   review criteria.
3. **Create the subcategory files.** Selectively reuse `spec.yaml`,
   `extra_requirements.yaml`, and required templates from the chosen example.
   Rewrite capability-specific content. Do not copy its outputs, logs, caches,
   or runtime configuration.
4. **Connect configuration.** Check distributions, preferences, output mappings,
   valid turn counts, and rules actually enforced by the runner. Explain each
   combination exclusion without weakening the target to improve success rates.
5. **Write templates.** Single-turn capabilities appear in Description, Plan &
   Review, Script Generation, and Script Review. Multi-turn capabilities appear
   in Description, Plan, and Prompt Generation, with Plan Review, Prompt Review,
   and Video Review checking them. Filenames follow the loader; module names
   follow the paper.
6. **Define criteria.** Separate structural checks for the Validator from
   semantic decisions the Director makes using Reviewer evidence. Define
   individually decidable rubric requirements.
7. **Validate and document.** Run applicable checks below and update the taxonomy,
   relevant README, and real consumers of static subcategory lists. Adding a
   subcategory alone does not authorize changing formal datasets, the paper, or
   running experiments.

Prefer configuration and templates for extensions. If the framework needs a
new mechanism, explain the smallest reusable change and compatibility effects.
Keep the capability intact rather than hard-coding its name in core branches.
Follow the target repository's applicable review and commit conventions.

## Validation

**Start offline.** Load specifications, check template syntax and actual rendering
contexts, calculate the assignment space, preview seeded sampling, and check
required fields and rules. See the configuration reference. File creation is not
evidence that these checks passed.

**Compare capability cases.** Consider a positive example and a counterexample
that changes the decisive condition:

- If removing the cue or required history leaves the answer recoverable, check
  for textual answer leakage, common-knowledge shortcuts, or a shifted target.
- If changing an object, speaker, orientation, emotion, or state should change
  the answer, the criteria must detect that change.
- Fix vocabulary coverage when valid English or Chinese expressions are
  rejected; preserve the underlying capability requirement.
- If evidence requires actual audio or video, mark a static plan as awaiting
  media verification.

**Integrate within the authorized scope.** `--test-run` may call a text backend
and should not be described as offline. When real synthesis is requested, use a
small batch, inspect each clip, and check that replies and rubrics follow the
observed media. Creating a subcategory does not authorize large-scale generation;
continue directly when integration requirements and budgets are already clear.

## Completion Criteria

- The current framework discovers the subcategory and accepts its specification
  and template interfaces.
- Its capability, neighboring boundaries, evidence sources, and characteristic
  errors are explicit.
- Diversity settings are feasible; nominal coverage is distinguished from
  capability coverage in actual media.
- Templates contain no stale base-subcategory identity, incompatible requirements,
  or unavailable variables.
- Reply and rubric fields, language rules, and tier conventions remain compatible.
- Multi-turn evaluation uses the supplied intermediate replies as fixed context
  and generates and scores only the final response.
- Validation reports actual results and identifies unperformed media or semantic
  checks.

Deliver the subcategory path, design decisions, changed files, and validation.
Distinguish candidate examples, completed synthesis, and human-confirmed formal
data. A new subcategory does not automatically increase the published benchmark's
reported size.
