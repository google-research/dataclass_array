---
name: review-task
description: Review a completed task within a project.
---

You have been assigned to review a completed task. You will receive:

-   **`<project_dir>`** — the project directory (contains `PROJECT.md`)
-   **`<todo_file>`** — the path to the TODO file describing the task
-   **`<workspace>`** - Set this as your cwd for all commands to be in the right
    folder!!

## Procedure

### Step 1: Read context

1.  Read `<project_dir>/PROJECT.md` for architecture, conventions, and
    verification steps.
2.  Read `<todo_file>` for the task specification and scope.
3.  Read these skills before reviewing:
    -   `api-design`
    -   `swe-style-guide`
    -   `test-design`

### Step 2: Verify

1.  Check for untracked files: `hg status` — look for `?` lines. If found, `hg
    add` them and re-upload the cl.
2.  Run `hg fix` and re-upload the cl.
3.  Run the verification steps from `PROJECT.md`. Do they pass?

### Step 3: Review

1.  Did the implementation solved the issue ?
2.  **Docs** — if the task adds user-visible behavior, was the documentation
    correctly updated ?
3.  Is there future cleanups/refactoring which cannot be implemented in this cl
    ? If so, use `write-todo` to track it and amend it and upload the cl.

### Step 4: Send review

1.  Fetch the CL with `fetch_changelist` (diffs + comments).
2.  Post your comments via the `critique` skill. The author cl is another agent,
    so no need to validate them before sending.

### Step 4: Report

Report back to the project lead with:

1.  **Verification result** — pass or fail (with error output if fail).
2.  **Issues found** — list each issue with the file and what needs to change.
3.  **Verdict** — `approved` or `needs changes`.

## Rules

-   Do NOT modify files in `<project_dir>/_agents/todos/` — the project lead
    handles task doc updates.
-   Do NOT fix issues yourself — report them. The lead decides whether to send
    fixes back to the original agent or handle them differently.
