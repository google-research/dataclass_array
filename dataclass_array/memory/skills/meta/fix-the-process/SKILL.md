---
name: fix-the-process
description: >
  Always fix the problem **and** the process that allowed it. Use when the user
  points out a mistake, gives feedback, or asks a question that reveals a gap.
---

## The rule

Every time the user asks something, gives a task, points to an issue, or
corrects a mistake, follow this 2 steps process:

1.  Fix the problem
2.  Fix the meta-problem (following the `update-skills`)

IMPORTANT: SKILLs are your own cognition so an error here can have significant
consequences. Always apply all steps thoughtfully, even for small fixes. Do
not rush the `update-skills` diagnosis.

### 1. Fix the problem

Solve the immediate issue.

### 2. Fix the meta-problem

Make sure the issue does not happen again by following the `update-skills`
process for this.

**Example 1:** User says "you forgot to amend the files to the cl."

-   Step 1: amend the files.
-   Step 2: investigate *why* — the review cl skill forgot this instruction →
    fix the skill.

**Example 2:** An external user asks "does the library support X?"

-   Step 1: answer their question.
-   Step 2: ask *why didn't they find the answer themselves?* → documentation
    was missing → create the docs. The question itself is the bug report.

**This applies to every user interaction — even the first message of a
conversation, and even when the user didn't explicitly ask for a meta-fix. It
also applies when helping *other* users — their questions and confusion are
signals of missing processes, docs, or error messages.**

**Step 2 is recursive.** When a user correct you on the meta-fix, that
correction itself requires both steps:

1.  Fix the meta problem (following the `update-skills`)
2.  Fix the meta-meta problem (following the `update-skills`)

You MUST follow `update-skills`.

## Related

-   `update-skills` — how to place fixes in the skill system
-   `write-skill` — creating a new skill
-   `self-improvement` — broader reflection and learning
-   `multi-level-thinking` — applying the fixes and lessons at multiple levels
