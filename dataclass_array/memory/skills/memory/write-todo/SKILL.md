---
name: write-todo
description: >
  Rules to write todos/ correctly. Use before writing anything in todos/, or
  when user report a todo.
---

## Structure

TODOs live in `<project>/_agents/todos/`:

-   One file per issue with YAML frontmatter (`type: todo`) and body content.
    See template section for format.
-   **Create tasks** when the user identifies a problem or feature. Each task
    describes the *problem and context* — not a solution. Include pointers to
    relevant files. Use the structured template from
    [write-todo/templates.md](../write-todo/templates.md).
-   **Numbering is permanent.** Never reuse or change task numbers. New tasks
    always get the next sequential number.

## When to add a TODO

When you discover something worth fixing but out of scope for the current task:

1.  **Find the home.** Choose the most specific `<project>/_agents/todos/` directory
    (local to the project if possible, otherwise global). Create it if it
    doesn't exist.
2.  **Create the fragment.** Follow [use-memory](../use-memory/SKILL.md) for
    placement rules and template section for the format.
3.  **Set the author.** Use `author: user` when the *user* explicitly asks to
    create a TODO. Use `author: agent` (the default) when you discover the issue
    yourself.

## Split TODOs in sub-todos

When the feature is large and contains multiple sub-tasks, those tasks should
each get their own separated TODO and referred in the main todo. If you're
creating the split, use `author: agent` for the sub-todos. If the split is given
by the user (directly or indirectly), use `author: user`.

## Content

Each TODO may be assigned to a **separate agent with no shared context**.
Write accordingly:

-   **Self-contained.** The agent should be able to complete the task by reading
    this TODO and explore the codebase on its own. Don't assume knowledge from
    other TODOs or conversations.
-   **Problem and context, not solution.** Describe WHAT is needed and WHY. Link
    to design docs for the HOW. Don't duplicate designs into TODOs — the
    codebase and designs evolve; the TODO should point to the source of truth.
-   **Keep it high level.** The codebase may change between when the TODO is
    written and when it's executed. TODO should describe the problem but APIs,
    internal details, function names,... should be let to the agent to decide.
-   **Don't prescribe implementation.** No code snippets, no "create file X with
    content Y." The agent makes local implementation decisions.
-   **Don't list individual files.**: Agent can decide the layout, or find the
    existing layout in the architecture docs.
-   **Keep it short.** The agent can figure out the details itself.

## Batch decomposition

When splitting a large feature into multiple TODOs, use `write-todo-plan` for
the decomposition process (granularity, dependency graph, parent TODO
structure).

## Template

```markdown
---
title: {Short Title}
type: todo
status: new
author: {user|agent}
priority: {p0|...}
difficulty: {easy|medium|hard}
project: {slash/delimited/project}
depends_on: []
date: {YYYY-MM-DD}
---

# {NNN} — {Short Title}

## Problem

{Description of the problem and context. Focus on:
- WHAT is wrong
- WHY it matters

Do NOT include:
 HOW to fix it.

Don't prescribe the implementation! This is NOT a design doc.
}

## Context

{
Where this problem comes from to track the source (e.g. link to a chat, conversation with user,...)

Do NOT include any technical details, this is just to track the history.
}

## Reproduction

{If relevant, how to reproduce the issue (e.g. steps, snippet,...)}

## Resolution

*Pending.*
```
