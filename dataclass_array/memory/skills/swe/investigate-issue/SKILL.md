---
name: investigate-issue
description: >
  Investigate and fix broken tests, auto-rolled back CLs, or production issues.
  Use when debugging failures — reproduce first, theorize later.
---

## Process

To investigate an issue, follow these gates in order:

1.  **Symptom** — note the exact reported error.
2.  **Reproduce** — through the actual entry point (CLI, binary, `bazel run`),
    not just unit tests. **Do this before reading any source code.** Read only
    what is needed to find the repro command (e.g. the bug report, SKILL.md).
3.  **Root-cause** — find the bug, then ask: "does this bug produce *that*
    symptom?" If you can't connect them, keep looking.
4.  **Fix.**
5.  **Verify end-to-end** — through the same entry point used to reproduce.

### Hypotheses to consider

-   **Released binary divergence.** MPM / CLI binaries (`/binfs/...`,
    `/google/bin/releases/...`) are released periodically via `rapid` and may
    lag behind the workspace source code. When a binary doesn't behave as the
    source suggests, try `bazel run //path/to:target` to rule out version
    mismatch.

### Misc

-   Use the personal folder (see `AGENTS.md` Environment) for temporary scripts,
    one-off tools, or repro harnesses needed during investigation.

## Related

-   `fix-the-process` — after fixing the bug, fix the process that allowed it
-   `test-driven-development` — write a failing test before applying the fix
-   `task-lifecycle` — investigation fits into the broader task workflow
