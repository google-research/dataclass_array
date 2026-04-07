---
name: write-skill
description: >
  Guidelines for writing effective skills. Use when creating a new skill.
---

If you read this before `update-skills`, STOP. First read `update-skills` to
understand the mental model of good skills.

## Naming and discoverability

-   **Name for the user's mental framing**, not the technical content. A skill
    named `investigate-issue` gets triggered by "debug this failure"; a skill
    named `solve-task` does not — even if it contains debugging rules.
-   **Use action verbs for action skills**. If a skill performs an action, name
    it with an active verb (e.g., `apply-review` instead of `review`) to avoid
    ambiguity with nouns or standard processes.
-   The `name` and `description` fields are the discovery mechanism. If they
    don't match how the task is described, the skill won't fire.
-   **Prefer skills over workflows** — skills are auto-discoverable in every
    conversation; workflows require knowing the filename.

## Scope

-   **One concern per skill.** If a skill covers two unrelated task types, split
    it. Each skill should trigger for a clear, distinct situation.
-   **Every skill has a trigger.** Never label a skill "always on" or
    "background" — that just means the trigger is unnamed, which is worse than
    narrow. Name the concrete conditions that activate it.
-   A skill can grow over time — add lessons as you learn them.
-   **Entry-point vs sub-skill.** Only entry-point skills belong in `AGENTS.md`.
    Sub-skills (reached only through another skill's cross-reference) must not
    be registered there — the parent's cross-reference is the edge that makes
    them reachable.

## Structure

Make sure you read and understand the `update-skills` to understand the right
mental model to write good skills. Do NOT start writing a skill before reading
`update-skills`.

-   Lead with the core principle or decision rule.
-   Follow with concrete steps or procedures.
-   Don't duplicate the trigger. The YAML `description` is the discovery
    mechanism. Don't add a separate "When to trigger" section.
-   Don't repeat rules from referenced skills. Don't restate its rules in your
    skill.
-   Keep rules concise. State the instruction; omit justification when the rule
    is self-evident.
