---
title: TODO Fields
type: reference
description: Allowed values for status, priority, and project fields in todo fragments
---

# TODO Fields

## `status`

-   **`new`** — Not started.
-   **`in-progress`** — Actively being worked on.
-   **`done`** — Completed with resolution.
-   **`descoped`** — Removed from scope with rationale.

## `priority`

-   **`p0`** — Broken or missing essentials. Blocks basic usage.
-   **`p1`** — Important capability gap. Blocks meaningful use.
-   **`p2`** — Noticeable improvement. Worth prioritizing.
-   **`p3`** — Advanced or power-user. Not blocking.
-   **`p4`** — Infrastructure, DX, or minor polish.
-   **`p5`** — Nice-to-have, cosmetic, or speculative.

## `project`

Slash-delimited tree. `external/` prefix for projects we don't own.

-   `kauldron/kontext` — Kontext key resolution, NNX wrappers.
-   `external/agents/critique` — Critique CLI bugs.
-   `agent/memory` — This memory system itself.

## `difficulty` (optional)

-   **`easy`** — Mostly configuration or documentation.
-   **`medium`** — Moderate code changes, clear scope.
-   **`hard`** — Significant complexity, multiple modules, or integration work.

## `depends_on` (optional)

List of task numbers this task depends on. Example: `[001, 002]`.

## `author`

-   **`user`** — Created by the human. **Takes precedence** over `agent` at
    the same priority level.
-   **`agent`** — Created by an agent. **Default when omitted.**
