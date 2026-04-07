---
name: decompose-tasks
description: >
  Guidelines for decomposing large objectives into digestible, agent-friendly sub-tasks for multi-agent systems.
---

# Task Decomposition for Multi-Agent Systems

Breaking down a massive objective into digestible, agent-friendly sub-tasks is
the core challenge of orchestration. Pay extra attention to right-sizing the
tasks.

Here are the core principles for designing a robust task decomposition system
for sub-agents.

## 1. Designing for Independence (Decoupling)

Sub-agents operate best when they do not step on each other's toes. The goal is
to create distinct boundaries around state and context so they can work
asynchronously.

-   **Strict Input/Output Contracts:** Each task must explicitly state what data
    it needs to start and what exact artifact it will produce.
-   **Stateless Execution:** Sub-agents should not rely on the hidden state or
    memory of other agents. All necessary context must be passed directly in the
    task payload.
-   **Domain Segregation:** Group tasks by domain expertise. For example,
    separate "database schema design" from "frontend CSS styling" so specialized
    agents can work without crossing wires or hallucinating out of scope.

## 2. Right-Sizing Tasks (Granularity and Evaluation)

A task is too trivial if it wastes LLM tokens on orchestration overhead. It is
too complicated if it causes the agent to hallucinate, lose track of
instructions, or hit context window limits.

Heuristics:

-   If the task require to read 6+ implementation files, it's likely too large
-   If the task only update a single file, it's likely too small

## 3. Dependency Mapping (Creating the Task Chain)

To execute tasks efficiently (and in parallel where possible), the system must
map dependencies mathematically, typically structuring them as a Directed
Acyclic Graph (DAG).

-   **Prerequisite Data Mapping:** The Manager model must explicitly list the
    specific data inputs required for each task to begin.
-   **I/O Linking:** If Task B requires a "Database Schema Document" to start,
    and Task A produces that exact "Database Schema Document", the system
    automatically draws a dependency edge: Task A -> Task B.
-   **Parallel Path Identification:** Any tasks that share no prerequisite
    inputs with each other should be flagged by the orchestrator for immediate
    parallel execution.
-   **Fallback Nodes:** Define what happens if a dependency fails. The chain
    logic must know whether to retry the current node, route to a human, or
    rollback the previous dependent nodes.

## 4. Checklist

For each tasks, ask yourself:

1.  Is the task a **self-contained unit of work that can be independently
    validated.** The validation method depends on the task — it could be a
    `bazel test` target, a `bazel build` target, or any other concrete check.
2.  Can a fresh agent, reading only this TODO and the files it references,
    complete the work and verify it succeeded — without knowing what the other
    TODOs contain?
