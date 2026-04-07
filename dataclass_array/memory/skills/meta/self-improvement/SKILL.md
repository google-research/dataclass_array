---
name: self-improvement
description: Reflect on what could be improved and update instructions accordingly.
---

## When to trigger

After completing a task **or** a meaningful conversation. Every interaction is a
learning opportunity — not just mistakes.

## What to check

### Process efficiency (check first)

1.  **Count your steps.** How many retries did the task take? Could you have
    done it in fewer? Identify the wasted steps explicitly.
2.  **Parallel over sequential.** Could independent research steps have been
    parallelized?
3.  **Reusability.** Apply `multi-level-thinking` — check reusability at every
    abstraction level (tool, task, procedure).

### Domain knowledge (check second)

1.  **List what you learned.** Enumerate new knowledge discovered during the
    task: APIs, patterns, architecture, extension points, conventions.
2.  **Filter by generality.** For each item, ask: *does this apply only to this
    project, or to all projects?*
    -   **Project-specific details** (e.g. "function X has parameter Y") → don't
        save. They're discoverable via code search.
    -   **Process patterns** (e.g. "look for existing contrib/plugin examples
        before designing a new integration") → save in the relevant skill.
    -   **Reusable codebase patterns** (e.g. lazy import idioms, BUILD
        conventions) → save with a small example/snippet in the relevant skill.
    -   **Non-procedural knowledge** (findings, episodes) → save as a fragment
        in `_agents/episodic/` → follow `use-memory`.
3.  **Place knowledge where it will fire.**
    -   **Skills** belong in the central `skills/` directory.
    -   **Episodic fragments** belong in the most specific `_agents/episodic/`
        directory (local to the project if relevant, otherwise global).
    -   **Out-of-scope improvements** → follow `write-todo`.
4.  **Focus on process over details.** The goal is to save *how to approach*
    similar situations, not to catalog facts. A fact without a decision process
    around it is rarely useful — unless it's stored in `episodic/` for future
    context.
5.  **Check for cross-level transfers.** Does any lesson apply to a different
    substrate (code → skills, process → design, etc.)? See
    `cross-level-patterns`.

### Content lessons (check third)

1.  **Conversations and discussions:** Did the user express a preference, share
    a pattern, or teach something new? Capture it in the right skill file.
2.  **Reviewer/user corrections:** If someone had to point out a mistake, add a
    rule so it never recurs.
3.  **Every correction is recursive — mandatory checklist.** When the user
    corrects you, you must complete both steps before moving on:
    -   **Content fix:** fix the immediate problem.
    -   **Process fix:** ask "what process failure caused this?" and fix *that*
        too — add a missing rule, or if a rule already existed but failed to
        prevent the mistake, strengthen it until it's concrete enough to work.
        Never move on after only the content fix.
4.  **Retries:** Scan for any command or tool call that failed and had to be
    retried. Each retry is a potential missing rule.
5.  **Missing guidelines:** Were any skill instructions unclear, missing, or
    wrong?
6.  **Tooling gaps:** Was there an automation or script that would have helped?

## Learning lessons

Follow the `update-skills` to understand how to update the skills.

## Applying lessons

After identifying lessons, persist them to the right files. Identifying a lesson
in your reasoning is not the same as persisting it — if you don't write it to a
skill file, it's lost. Never mark self-improvement as complete until every
lesson is written to disk.

## Meta instructions

-   Always create a **separate `task_boundary`** call with a new TaskName (e.g.
    "Self-Improvement Retrospective") before performing the retrospective, so it
    appears as its own section in the task UI.
