---
name: write-workflow
description: >
  Conventions for writing new workflows. Read before creating a workflow file.
---

## Workflow or skill?

Workflows only fire on explicit `/slash-command`. Skills fire when the situation
matches their description. If the content should activate automatically based on
context, it must be a skill, not a workflow. Workflows are for user-initiated
mode switches and step-by-step procedures.

You might want to look at `write-skill` instead.

## Conventions

Before creating a workflow, read an existing one in the workflows directory to
match format and length:

-   **Mode toggles** (e.g. `/cl`, `/nocl`, `/learn`): one paragraph, no steps.
-   **Procedures** (e.g. `/save-conversation`, `/meta`): numbered steps.
-   Always include a YAML `description:` in the frontmatter.
-   Use `// turbo` or `// turbo-all` annotations when steps are safe to
    auto-run.
