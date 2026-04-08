# Project Agent Rules

## Scope
- This file applies only inside `/home10T/bzx/workspace/kvpress-study`.
- The local skill `experiment-results-hygiene` is project-scoped and must only be used in this repository.
- Do not assume this skill exists or is applicable when working outside this repository.

## Local Skill Activation
- Activate `experiment-results-hygiene` only when the task is about organizing, regrouping, indexing, or documenting experiment outputs for this repository.
- The skill definition lives at `/home10T/bzx/workspace/kvpress-study/.codex/skills/experiment-results-hygiene/SKILL.md`.
- When this repository is the active working context and such a task appears, read that file directly and follow it as the local skill source.
- Treat all directory conventions in that skill as `kvpress-study`-specific, especially:
  - `figure/`
  - `figure/experiments/`
  - `evaluation/results/`
  - `evaluation/results/experiments/`
  - `evaluation/results/ad_hoc_baselines/`
  - `note/`

## Safety
- Before reorganizing experiment outputs, preserve raw artifacts and keep paths reproducible.
- If a task would move or rename result directories referenced by scripts, update the affected scripts or docs in the same change.
