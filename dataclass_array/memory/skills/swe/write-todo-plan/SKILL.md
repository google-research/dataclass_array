---
name: write-todo-plan
description: >
  Split a design doc into small, self-contained sub-tasks for agent execution.
  Use when user ask you to create an implementation plan.
---

# Decomposing a design into TODOs

Given an approved design doc, produce a parent TODO + self-contained sub-task
TODOs in the project's `todos/` directory. Each sub-task will be assigned to a
**separate agent with no shared context** — this constraint drives every
decision below.

## 1. Decompose in sub-tasks TODO

Follow `decompose-tasks` rules to decompose the tasks. This is the most
important step.

## 2. Write the parent TODO

The parent TODO is the coordination surface. It must contain:

1.  **Problem and context** — what the design achieves, link to the design doc
2.  **Sub-task list** — numbered references to each sub-task TODO. Use
    `decompose-tasks` for this.
3.  **Dependency graph** — which sub-tasks depend on which (text or mermaid)
4.  **Parallelism waves** — which sub-tasks can run concurrently

The parent does NOT contain implementation details.

## 3. Write the tasks as TODOs

Use `write-todo` for file format and placement rules to write the output.

Each sub-task describes the **problem and context** — not a solution. The agent
decides the implementation. Link to the design doc instead without duplicating
the content.

## Related

-   `write-todo` — file format, placement rules, numbering
-   `feature-design` — the design phase that precedes this skill
-   `distribute-todos` — multi-agent coordination that consumes these TODOs
