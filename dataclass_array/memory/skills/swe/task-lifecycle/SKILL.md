---
name: task-lifecycle
description: High-level rules for completing a task end-to-end.
---

## Lifecycle

Every coding task follows this cycle:

### 1. Solve the task

#### 1.1 Task workflow

Pick the right approach based on the task type:

-   **Designing a complex feature** → follow `feature-design` first to validate
    the user-facing API. Then proceed to implementation.
-   **Distributing a batch of todos** → follow `distribute-todos` to
    partition work, coordinate subagents, and merge progress.
-   **Implementing a feature or change** → follow `test-driven-development`.
-   **Moving, copying, renaming, or deleting files** → follow `refactoring`
    before any file operation.
-   **Debugging a failure or investigating a rollback** → follow
    `investigate-issue`.

#### 1.2 Add documentation

-   **Document new features.** When adding a new capability, update
    documentation (docs, docstrings, README, CHANGELOG) as part of the
    implementation — don't defer to a separate CL → follow `doc-style-guide`.

### 2. Self-improve (first pass)

Invoke the **self-improvement skill** (`self-improvement/SKILL.md`). Reflect on
the code and instructions while the work is fresh.

### 3. Create / update the CL

Follow the **create-cl skill** (`create-cl/SKILL.md`) to commit, format the
description, and upload.

### 3b. Close the loop

Reply to whoever raised the issue (CL comment, chat message, bug, etc.) with a
link to the fix CL.

### 4. Review loop

When the reviewer leaves comments:

1.  Invoke the **apply-review skill** (`apply-review/SKILL.md`) to read comments, apply
    fixes, build/test, and reply.
2.  Follow the **create-cl skill** to amend and upload the new patchset.
3.  Repeat until all comments are resolved and the CL is approved.

### 5. Self-improve (final pass)

After the review loop is complete, invoke the **self-improvement skill** again.
Incorporate any lessons learned from review feedback (e.g. patterns the reviewer
flagged, conventions that were missed).

## Rules

*   Never skip any step — every coding task goes through the full cycle.
*   If the task is purely exploratory (no code changes), skip steps 3–5 but save
    research findings following `use-memory` — knowledge that isn't persisted is
    lost.
*   **Bootstrap `task.md` with lifecycle steps.** When creating the initial
    `task.md`, add all lifecycle steps (1.2 docs, 2 self-improve, 3 create CL)
    as explicit todo items. This prevents skipping steps that aren't on the
    immediate execution path.
*   **Skills listed in `task.md` are executable references, not labels.** When
    you write `following skill-name` next to a task, you are committing to
    re-read that skill and follow its steps. If the skill has sub-steps (e.g.
    per-batch output files in `batch-processing`), those sub-steps must appear
    as checkboxes in `task.md`. Listing a skill without executing it is the same
    as calling a function and ignoring its return value.

## Summarization

Instructions for the summarizer agent: When `summarizing this work`, the agent
is gonna lose all skills context. You MUST add in your summary something like:

```
Before continuing the task, you **MUST** start by applying `refresh-context`
skill. This is critical to correctly reload the skills and context and not miss
out critical informations.
```
