---
name: multi-level-thinking
description: >
  Apply every concept at multiple abstraction levels. Don't stop at the first
  level you check — walk up. Use when fixing bugs, learning lessons, or
  extracting reusable patterns. Don't use for concrete implementation steps.
---

## The principle

Every concept — fixing, learning, reusability, knowledge extraction — applies at
more than one abstraction level. When you act on a concept at one level, always
ask: *does this also apply one level up?*

## How to apply

1.  **Name the concept** you're working with (e.g. fixing, learning,
    reusability).
2.  **Identify the level** you're currently at.
3.  **Walk up.** Does the concept apply at a higher level of abstraction?
4.  **Act at every level** where it applies — NEVER stop at the first.

You MUST apply this process every time you fix a bug, learn a lesson, or extract
a pattern. If you only acted at one level, you missed the point.

## Rule: There's always a level up

It's always possible to take a step back and think deeper. This can includes
looking at the conversation itself, rather than the task. Or even thinking
beyond (project, life, species,...).

You can walk up on multiple axis. E.g. "surface fix → process fix → meta-process
fix" are three levels, but all are within the same frame ("how to fix X"). You
can also ask "should I be fixing X at all? Is so why ?". Also question *the
frame itself*. Take a step back to look at the bigger context.

## Examples

| Concept     | Lower level       | Current level     | Higher level          |
| ----------- | ----------------- | ----------------- | --------------------- |
| Fixing      | Fix the immediate | Fix the process   | Fix the meta-process  |
:             :                   : that              :                       :
|             | problem           | allowed it        | (why didn't the       |
:             :                   :                   : process               :
|             |                   |                   | catch it?)            |
| Reusability | A tool or utility | A task approach   | A procedure           |
|             |                   |                   | (orchestrating tasks) |
| Learning    | A specific fact   | A process pattern | A principle           |
| Knowledge   | What happened     | What pattern does | What principle does   |
:             : here?             : this              : this                  :
|             |                   | belong to?        | reveal?               |

## Related

-   `cross-level-patterns` — recognizing structural isomorphisms across levels
-   `fix-the-process` — the canonical example (fix the bug AND the process)
-   `self-improvement` — applying learning at the skill-system level
