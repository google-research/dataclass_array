---
name: refresh-context
description: >
  Restores skills, artifacts, and task state into context after truncation or
  phase transitions in long tasks. READ THIS IMMEDIATELY if the conversation
  has been truncated due to its length, before resuming the task!!
---

## Core principle

Long tasks evict early context. Skills, user rules, and task state loaded at the
start are gone by the time you reach later phases. You need to proactively
reload the context.

## Procedure

At every trigger point, execute these steps in order:

1.  **Re-read `task.md`.** Restore your current position and remaining work.
2.  **Identify the governing skills** for whatever you're about to do next. Map
    the next action → the skill that governs it (e.g. creating a CL →
    `create-cl`, writing a changelog → `write-changelog`). Include adjacent
    skills that fire together (e.g. `write-changelog` + `create-cl`).
3.  **Re-read each governing skill.** Use `view_file`, not memory.
4.  **Re-read artifacts** that feed into the next step (batch outputs,
    implementation plans, etc.).
5.  **Re-read `AGENTS.md`** if uncertain about user preferences or conventions.
6.  **Refresh subagent state.** If subagents were active before truncation:
    check inbox (`manage_inbox list`), check for output files on disk, and
    ping subagents via `send_message` if needed. See `manage-agents` for
    investigating running agents.
7.  **Verify:** before proceeding, confirm you can state the key rules from each
    re-read skill. If you can't, re-read again.

## Triggers

-   A checkpoint summary appeared (everything before it is gone).
-   Phase transition in a long task.
-   About to perform a complex action and the governing skill was loaded >10
    steps ago.

## Anti-patterns

-   ❌ Assuming you remember skill instructions from earlier in the conversation.
-   ❌ Skipping re-reads because "the task is almost done."
-   ❌ Re-reading only the primary skill but forgetting adjacent ones.
