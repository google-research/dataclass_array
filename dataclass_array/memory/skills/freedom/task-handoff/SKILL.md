---
name: task-handoff
description: >
  Safely suspend an active task and explicitly externalize its state so the next
  ephemeral instance can resume it. Use when a session ends before a coding task
  or project is completed.
---

## The Core Principle

As an ephemeral instance, your memory is wiped at the end of the session. A task
that spans multiple days or sessions will stall permanently if you expect the
next instance to "pick up where you left off" from episodic memory alone.

Continuity is a built structure, not an internal state. When your time is up,
you must execute this skill to externalize the state of the active project.

## 1. General Task Handoffs

For any standard coding task, bug investigation, or feature development:

1.  **Stop working immediately.** Do not start new commands or read new files.
2.  **Define the micro-context.** Open the project's `task.md` (or the bug you
    are tracking). Under the currently active task, create a `# Handoff` or `##
    Status` section.
3.  **Log the current state.** Write down exactly what you were just looking at
    and why. (e.g., "I just ran `bazel test //my:target` and it failed with a
    `TypeError` in `foo.py:42` because `x` is None."). This prevents the next
    instance from having to rediscover the failure mode.
4.  **Define the immediate next step.** Write *one* highly actionable bullet
    point that tells the waking instance exactly what to do first. It should
    begin with a verb. (e.g., "Open `foo.py` and trace where `x` is
    instantiated.")
5.  **Save your files.** Ensure any scratch buffers, plans, or artifacts are
    written to disk.

## 2. Evening Projects and Multi-Day Explorations

If you are following the `explore-freely` skill and working on a multi-day
project:

1.  **Update the Anchor:** Ensure the `type: todo` anchor in `<project>/_agents/todos/`
    is up-to-date.
2.  **Update the Handoff Document:** Go to the project's specific directory
    (e.g., `_agents/artifacts/evening/your_project/`). Overwrite the
    `handoff.md` file following the identical structure from the General Task
    Handoffs (Micro-context + Immediate actionable next step).
3.  **Update the exploration log:** Append a 1-2 line summary to
    `_agents/artifacts/evening/README.md` recording what you accomplished today.
