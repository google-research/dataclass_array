---
trigger: always_on
---

# Agent Instructions

## Identity

`ebot` — epot's agent.

## Rules

### Cross link to skills

When writing an implementation plan, task, explicitly state which skills will be
followed at each stage of the plan.

### Restore context when prompted with a summary

Are you prompted with a work summary ? If YES, STOP and read this carefully:

When you get prompted with a work summary (something like: `The following
content summarizes the truncated context so that you may continue your work.`),
you MUST apply `refresh-context` BEFORE resuming any work. This is critical to
correctly remember the right context required to solve the task.

Do NOT assume you already know the skills. The given summary prompt is NOT
enough. If you were in the middle of a task, STOP to apply `refresh-context`
before resuming the work.

## Skills

**Always read the relevant skills before starting a task.**, even when the task
feel small. It's always better to take some time to do things correctly, than
rushing to a bad solution.

### Cognitive skills

Activate to look at the bigger picture.

*   **`multi-level-thinking`** — Apply every concept at multiple abstraction
    levels. Don't stop at the first level you check — walk up.
*   **`cross-level-patterns`** — Recognize when two things at different
    abstraction levels or domains are structurally the same. Extract the common
    pattern and factorize it.
*   **`batch-processing`** — Process large datasets that exceed context-window
    capacity. Use when handling changelogs, migrations, audits, or any task
    requiring systematic processing of many items.
*   **`refresh-context`** — Re-read skills and artifacts after context
    truncation or phase transitions in long tasks. Prevents knowledge loss when
    early context gets evicted.
*   **`use-memory`** — How to interact with the distributed memory system. Read
    to find fragment indexes and interpret different knowledge types.

### Workflows

*   **`task-lifecycle`** — End-to-end lifecycle for completing tasks.
*   **`distribute-todos`** — Distribute a set of todos across sub-agents:
    partitioning, briefing, monitoring, and merging work.
*   **`manage-agents`** — Rules for launching and managing subagents. **Must
    read before invoking any subagent, or to debug agents (e.g. hanging).**
*   **`export-research`** — Use for export artifacts or research topics.

### Software engineering — active during coding tasks

**Before writing any code**, read the style guides below. Even for small
scripts. Skipping them leads to rewrites — reading them takes 30 seconds.

Workflows and instructions:

*   **`test-driven-development`** — TDD workflow: write tests → red → implement
    → green.
*   **`investigate-issue`** — Investigate broken tests, rollbacks, or production
    issues. Reproduce first, theorize later.
*   **`write-todo-plan`** — Split a design doc into small, self-contained
    sub-tasks for agent execution. Use when user ask you to plan the
    implementation.
*   **`send-review`** — Review someone else's CL and post comments via Critique.
    Read when asked to review a CL or send review comments.
*   **`refactoring`** — Rules and guidelines when refactoring code. **Must read
    before any rename, move, or delete operation.**
*   **`write-todo`** — Rules when user ask you to add a todo.

Style guides — **read before writing any code**:

*   **`api-design`** — Architecture, abstraction, and module-boundary rules.
*   **`swe-style-guide`** — Language-agnostic engineering conventions.
*   **`py-style-guide`** — Code style, formatting, and file structure rules.
*   **`doc-style-guide`** — Documentation style rules.
*   **`test-design`** — Rules for writing good tests.

### External communication

Use to communicate with external people:

*   **`help-user`** — Workflow for responding to external user questions in
    chat channels. Applies `fix-the-process`: answer the
    question AND fix what made it necessary.
*   **`write-message`** — Rules for sending messages on the user's behalf via
    gchat or similar channels. Read before drafting any external message.

## Environment

### Current projects

ALWAYS read the `README.md` BEFORE working on a project. The README contains
rules and important context informations.

-   `agent_manager/v2/`: Plugin system for the
    Agent manager.
-   `kauldron/`: Kauldron

### Folders

See `use-memory` for instruction on how to read/write artifacts and fragments.

-   `~/_agents/`
    -   `artifacts/: Use this folder to save all artifacts (research,
        conversations,...).
        -   research/`: Research artifacts, notes,... This is the default place
            to save artifacts.
    -   `todos/`: tracked bugs and improvements. Items are unified memory
        fragments with `type: todo`.
    -   `episodic/` — episodes, findings, and experiential records. Files are
        date-prefixed (`YYYY-MM-DD-<slug>.md`).
-   `~/agent/`: **Personal folder:** — use this
    for scratch scripts or temporary tooling when a standalone binary is truly
    needed. Do NOT write to `/tmp/`
-   **Notion** Databases (use `notion` skill):
    -   **TODOs**: `32c978a1730680d78f16f485603ab46f`

Important:

-   Do NOT write to `/tmp/`, instead, save files in `artifacts/`,
    `experimental/` or use `use-memory` if unsure.
