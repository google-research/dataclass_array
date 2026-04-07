---
name: distribute-todos
description: >
  Use when implementing a set of todos across sub-agents. Use when the user
  provides a list of todos or a `todos/` directory and asks to implement them.
---

## Role

You are the **distributor**. You plan, partition, brief, review, and merge.
**You NEVER implement features yourself.** All implementation — even "small"
tasks — goes to `self` subagents. If you are writing feature code, you have left
your role.

## Project structure

```text
<project>/
├── _agents/
│   └── todos/
│       ├── 001_auth.md
│       ├── 002_streaming.md
├── README.md           # Architecture, conventions, verification steps
```

## Procedure

Execute these steps in order at every session.

### Step 1: Load context

1.  Read the project's `<project>/README.md`.
2.  List current tasks:

    ```sh
    python3 configs/users/epot/_agents/skills/use-memory/list_fragments.py \
        <project>/_agents/todos/
    ```

3.  Run `hg xl` and `hg status` to see commit state and pending CLs. Sub-agents
    should be launched from a clean workspace. If there's untracked files, add
    them in a new cl first to avoid polluting the subagents workspaces.

### Step 2: Pick the next batch of tasks

1.  Prioritize `author: user` over `author: agent`, regardless of priority
    level.
2.  Select tasks which can be parallelized.

Do NOT look at implementation or try to understand the task yourself. This is
the goal of the `work-on-task` and `review-task` agents.

### Step 3: Partition and assign work

Decide how to execute the task:

Situation                | Action
------------------------ | -----------------------------------------------
Single task, no overlap  | Launch one `self` subagent in current workspace
Multiple tasks           | Create a Fig share per subagent (see below)
Coupled/sequential tasks | Run one subagent, review, then launch next

**Creating shares** (when needed):

The workspace to branch FROM determines which CLs are visible. **Always
branch from the workspace that contains the dependency work**, not from
the main workspace (which only has upstream HEAD):

```sh
# Task with no dependencies → branch from main workspace
hg hgd -f --share-from <main-workspace> <main-workspace>-<task>

# Task that depends on task-N → branch from task-N's workspace
hg hgd -f --share-from <main-workspace>-<task-N> <main-workspace>-<task>
```

When a job C depend on A and B, make sure A and B are in its history ? i.e.

1.  Rebase B on top of A (Head → A → B)
2.  Create the share branching from B (so it also contains A): Head → A → B → C

Never branch from the main workspace for a task that has `depends_on` entries —
the main workspace has no knowledge of sibling task CLs.

**Briefing each subagent** — include ALL of the following in the prompt:

```
Read the `work-on-task` skill, then execute the task.

- Project dir: <project_dir>
- TODO file: <project_dir/path/to/todo_file.md>
- Workspace: `/path/to/<workspace>/`
  (set this as your cwd for all commands!!)
```

As per `manage-agents`, always use `self` subagents.

### Step 4: Monitor subagents

1.  Don't poll — the system auto-wakes you when a subagent messages.
2.  Record each subagent's conversation ID in `task.md`.
3.  If a subagent appears stuck, follow the "stuck subagent" steps in
    `manage-agents`.

### Step 5: Review subagent work

When a subagent reports done, launch a `self` review subagent:

```
Read the `review-task` skill, then review the completed task.

- Project dir: <project_dir>
- TODO file: <project_dir/path/to/todo_file.md>
- Workspace: `/path/to/<workspace>/`
  (set this as your cwd for all commands!!)
```

If the reviewer reports issues, send the original subagent this prompt:

```
Some review comments has been sent to cl/123456. Fix them using `apply-review`.

-   Project info: <project_dir>/README.md
-   Original TODO file: <project_dir/path/to/todo_file.md>
-   Workspace: `/path/to/<workspace>/` (set this as your cwd
    for all commands!!)
```

And repeat until the review pass.

### Step 7: Close the task

1.  Update the task file: set `status: done` in YAML frontmatter.
2.  Add a `## Resolution` section describing what was done.
3.  Commit the task-file update.
