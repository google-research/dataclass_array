---
name: work-on-task
description: Execute a single task assigned by the project lead.
---

You have been assigned a task by the project lead. You will receive:

-   **`<project_dir>`** — the project directory (contains `README.md`)
-   **`<todo_file>`** — the path to the TODO file describing your task
-   **`<workspace>`** - Set this as your cwd for all commands to be in the right
    folder!!

## Procedure

### Step 1: Read context

1.  Read `<project_dir>/README.md` for architecture, conventions, and
    verification steps.
2.  Read `<todo_file>` for the task specification.
3.  Run `hg status` to ensure the workspace is clean and matches Piper before
    starting any code changes.
4.  Read these skills before writing any code:
    -   `api-design`
    -   `swe-style-guide`
    -   `test-driven-development`
    -   Any others relevant to the task.

### Step 2: Implement

1.  Follow `task-lifecycle` for the full implementation cycle.
2.  Stay within the task's scope. If you discover something out of scope, follow
    `write-todo` to file it — don't fix it now.

### Step 3: Verify

1.  Run the verification steps from `README.md`.
2.  Check for untracked files: `hg status` — look for `?` lines, `hg add` them.

### Step 4: Commit and report

1.  `hg commit -m "Task NNN: <short description>"`
2.  `hg upload`
3.  Report results back to the project lead.

## Rules

-   Do NOT modify files in `<project_dir>/_agents/todos/` — the project lead handles
    task doc updates.
-   One task = one CL. Don't bundle unrelated changes.
-   New files must be `hg add`-ed before committing.
