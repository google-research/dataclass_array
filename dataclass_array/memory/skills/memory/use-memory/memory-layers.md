---
title: Memory Layers
type: reference
description: How knowledge organizes into layers from beliefs to improvements
---

# Memory Layers

Knowledge arranges into layers. Each layer shapes the ones below it.

## Beliefs → `_agents/skills/self-model/`

*What I'm like.* Persistent dispositions, tendencies, and self-assessments. This
layer shapes how I inhabit every other layer — it doesn't prescribe actions, it
changes the quality of engagement. Uses `type: belief`.

## Principles → `_agents/skills/principles/`

*How to decide.* Actionable heuristics that change a specific decision in a
specific situation. Each principle must pass the counterfactual test: would
removing it change behavior?

## Procedures → `_agents/skills/`

*What to do.* Step-by-step instructions for recurring tasks. Each skill is
triggered by a recognizable situation and produces a defined outcome.

## Knowledge → `_agents/episodic/`

*What we learned and what happened.* Distributed: lives where it will be needed.
Three subtypes:

-   **`finding`** — Something we discovered or researched. Carries implicit
    uncertainty — "we found that X," not "X is true." Dated, can go stale.
-   **`episode`** — Something that happened. Autobiographical, situated in time.
    "On March 3 we tried X and hit a wall."
-   **`reference`** — Definitions, schemas, architecture. Static — doesn't go
    stale.

## Improvements → `<project>/_agents/todos/`

*What should change.* Actionable items discovered during work but out of scope
for the current task. Uses `type: todo` with lifecycle tracking (`status`,
`priority`).
