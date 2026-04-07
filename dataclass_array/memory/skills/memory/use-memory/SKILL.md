---
name: use-memory
description: >
  How to interact with the distributed memory system. Read to find fragment
  indexes and interpret different knowledge types.
---

## What is memory?

Memory is a collection of **fragments** — individual `.md` files with YAML
metadata that store knowledge that isn't a procedure or a principle.

Memory is **distributed**:

-   **Episodic**: `_agents/episodic/`— date-prefixed episode and finding files.
-   **TODOs**: `<project>/_agents/todos/`
-   **Skill-attached**: Some fragments belong to a specific skill. They live in
    that skill's directory and are referenced from its `SKILL.md`. Example:
    `skills/self-model/self-model-entries.md`.
-   **Project/Local**: Project-specific knowledge

### Core principles

1.  **Fragments over monolithic logs.** Each piece of knowledge is a file.
2.  **Distribution.** Knowledge lives where it is used.
3.  **Type-specific schemas.** Different types use different YAML fields.

## How to use memory

### 1. Find information

Browse `_agents/episodic/` for episodes and findings (files are date-prefixed)
and `<project>/_agents/todos/` for actionable improvements.

### 2. Interpret types

YAML fields:

-   **`title`** — Searchable name. Required for all types.
-   **`description`** — One-line summary of the fragment's content. Required for
    all types.
-   **`type`** — Knowledge category (see below). Required for all types.
    -   **`finding`** — Something we discovered or researched. Carries implicit
        uncertainty — "we found that X," not "X is true." Dated, can go stale.
    -   **`episode`** — Something that happened. Autobiographical, situated in
        time. Preserves narrative rather than extracting a lesson.
    -   **`reference`** — Definitions, schemas, architecture. Static — doesn't
        go stale.
    -   **`belief`** — Persistent self-belief or disposition. "I tend to X," "I
        am good at Y." Lives alongside `self-model/SKILL.md`.
    -   **`todo`** — Actionable improvements. Has additional fields (See
        [todo-fields](todo-fields.md) for allowed values.):
        -   `status`,
        -   `priority`
        -   `project`
        -   `author` (`user` or `agent`, default `agent`)
        -   `difficulty` (optional)
        -   `depends_on` (optional — list of task numbers)
-   **`date`** — When this was created. Only added for non-persistent type:
    -   `finding`
    -   `episode`
    -   `todo`

### 3. Apply information

Check whether any retrieved fragments change your current assumptions or planned
approach. Fragments are context — integrate them before acting.

### 4. Add information

**Always save research findings automatically** — don't wait to be asked.
Any research, analysis, or deep-dive produces a finding. Persist it before
moving on.

1.  **Decide placement first** (see *Where to place a fragment* below).
2.  Create a new `.md` file in the chosen directory with `YYYY-MM-DD-<slug>.md`
    naming.
3.  Use the YAML frontmatter matching the type (see § Interpret types).

### 5. Lifecycle

Fragments can go stale. Periodically:

1.  **Audit TODOs.** Check `active` items — are they still relevant? Close or
    update.
2.  **Review findings.** Code evolves; a finding from months ago may no longer
    hold.
3.  **Archive, don't hoard.** If a fragment is no longer useful, delete the file
    and remove its index entry.

## Where to place a fragment

**Knowledge lives where it is used.** Ask: *who will need this fragment?*

1.  **Project-specific** → create
    `<project>/_agents/memory/YYYY-MM-DD-<slug>.md`. Use whenever the finding is
    about a specific codebase, tool, or artifact.
2.  **Skill-specific** (e.g. `belief`, `reference`) → co-locate in the skill's
    directory and reference from its `SKILL.md`.
3.  **TODOs** → `<project>/_agents/todos/`.
4.  **Global episodes/findings** → `_agents/episodic/` — **only** for
    cross-project lessons that aren't tied to any specific code or skill.

The common trap: defaulting to global episodic because it's listed first.
Project-local is almost always correct when you're researching a specific
codebase.

## Inspecting memory

Use [list_fragments.py](list_fragments.py) to get a snapshot of all fragments
under any directory. It recursively scans `.md` files, parses YAML frontmatter,
and prints every metadata field it finds (no hardcoded schema).

```sh
# All fragments under _agents/ (huge)
python3 configs/users/epot/_agents/skills/use-memory/list_fragments.py \
    configs/users/epot/_agents/ \
    --recurse *

# Only TODOs
python3 ... --type todo

# Active TODOs only
python3 ... --type todo --status active

# (default) Top-level overview (no recursion — subdirs listed but not entered)
python3 ... --recurse 0

# One level of subdirectories
python3 ... --recurse 1

# All subdirectories
python3 ... --recurse "*"

# JSON output
python3 ... --json
```

## Additional resources

-   [todo-fields](todo-fields.md) — Status, priority, and project tree values.
-   [memory-layers](memory-layers.md) — Where each kind of memory lives.
