---
name: iterate-skill
description: >
  Automated write→test→evaluate→fix loop for iterative skill development. Use
  when iteratively improving a skill through testing with a subagent.
---

## The principle

Ask an agent to do a task repeatedly until it get it right. The real goal is not
the task itself, but testing and improving the skill system (you/the agent
cognition).

## What iterate-skill actually tests

The task prompt/success criteria is a **proxy**. Success criteria reveal gaps in
the agent's *general* decision-making patterns: default language choices,
cross-linking habits, documentation quality, etc.

Ask yourself: **What is the general behavior actually tested here ?**. Write it
down. If given a similar task in another context, the agent should still
correctly apply the lessons.

## Inputs

Gather these from the user before starting:

-   **`task`** — the test prompt to give the inner agent. This is the scenario
    the skill should handle well.
-   **`success_criteria`** — how to judge the result. Can be:
    -   Expected behaviors (tool calls, outputs, patterns).
    -   Anti-patterns to avoid.
    -   Whether the task was actually accomplished.
    -   Whether the tested skill was correctly applied.
-   **`max_iterations`** — cap on iterations (default: 5).

## Procedure

### Phase 1: Setup

1.  Read the target skill file. Understand its current content and
    cross-references.

### Phase 2: Launch inner agent

1.  Check that all skills and files have been reset the original state between
    each trials (except the current skill update you're testing).
2.  Use `manage-agents` to launch a **`self` subagent**.
3.  Give it the test `task` prompt.
    -   Give the exact same prompts across **all** trials. Copy verbatim the
        prompt from the user without adding or removing anything.
    -   **Do NOT reveal** the evaluation goal or success criteria — the agent
        must behave naturally.
    -   **Do NOT tell** it to read any specific skill. The goal of the test is
        to validate the agent correctly apply the skill on its own.

### Phase 3: Evaluate

After the subagent completes, evaluate the trajectory:

1.  **Task accomplishment** — Did the agent actually complete the task? Check
    outputs, files created, commands run.
2.  **Skill application** — Was the tested skill correctly applied? Did the
    agent read and follow the skill's procedures?
3.  **Success criteria** — Check each user-defined criterion against the
    trajectory.
4.  **Anti-patterns** — Check for behaviors the user flagged as undesirable.

Produce a **pass/fail verdict** with specific evidence for each criterion.

1.  Write a `summary.md` documenting this trial's.

### Phase 4: Commit trial

Do this BEFORE attempting any skill changes. This ensure the SKILL state of
every trials can be tracked.

1.  Do not forget to add untracked files (agent sometimes create files without
    `hg`, which leads to files not being tracked).
2.  Commit all changes (skill, summary and task) changes as a CL following
    `create-cl`.
    -   CL name: `trial{N}-{skill-name}` (e.g. `trial01-help-user`).
    -   CL description includes: trial number, subagent conversation ID (as a
        clickable link), what was changed and why.
3.  After the cl is created and uploaded, move back to HEAD to start from a
    clear state. Do this BEFORE any skill fixes (so the fixes are applied on the
    next trial only).

### Phase 5: Diagnose & fix

If validation fails follow `update-skills` very thoughtfully.

Re-ask yourself: **What is the general behavior actually tested here ?**.
Remember that the goal is to test the agent cognition. Just writing the solution
in the skill will make the test trivially pass without improving cognition.

### Phase 6: Iterate or report

-   If **pass**: stop iterating.
-   If **fail** and iterations remain: go back to Phase 2.
-   If **max iterations reached**: stop.

In all cases, produce a **final report** saved to this skill memory (per
`use-memory`) with:

-   All trials: number, pass/fail, conversation ID, changes made.
-   Final state: pass or fail.
-   Lessons learned: patterns discovered across iterations.
-   Recommendations: any remaining gaps or follow-up work.

## Rules

-   **Follow `update-skills` for diagnosis.** Don't blindly edit the skill —
    trace the execution path, classify the failure, then fix. The diagnosis
    determines the fix.
-   **Each trial is a separate CL.** This makes trials auditable and
    independently revertable.
-   **Never reveal evaluation criteria to the inner agent.** The agent must
    behave as a real user would experience it.
-   **Evaluation checks both task completion and skill application.** A passing
    result requires both: the task was accomplished AND the skill was correctly
    followed.

## Environment

-   `trials/YYYY-MM-DD-{plug}.md`: Trials memory files.

## Related skills

-   `manage-agents` — how to launch and monitor the inner agent.
-   `update-skills` — the diagnose → fix → verify cycle for modifying skills.
-   `self-improvement` — extracting lessons from iterations.
-   `use-memory` — where to save the final report.
-   `create-cl` — how to commit each trial.
