---
title: Cross-Level Trigger Analysis
type: finding
date: 2026-03-05
description: When cross-level pattern recognition should fire during workflows
---

# When does cross-level pattern recognition fire?

## The question

Before designing the skill, we need concrete examples of: *I'm in the middle of
doing X, and at that moment, I should notice a cross-level pattern.* What is X?
What does "noticing" look like? What action follows?

## Candidate trigger moments

### Trigger 1: During self-improvement retrospective

**Situation:** I just completed a coding task. In the self-improvement phase,
I'm listing what I learned.

**Pattern recognition moment:** I realize that the lesson I learned in code
(e.g., "these two functions were duplicates — I should have extracted a shared
utility") is structurally identical to something in the skill system (e.g., two
skills encode the same procedure — I should extract a shared principle).

**Action:** Apply the lesson not just to code, but also to
skills/processes/docs.

**But:** This is already partially covered by `multi-level-thinking` — "check
reusability at every abstraction level." The difference is that multi-level
thinking says "walk up from code to process," while cross-level recognition says
"the *technique* I used for code (extract shared function) is the *same
technique* for skills (extract shared principle)." Multi-level transfers the
*concept*, cross-level transfers the *procedure*.

### Trigger 2: During code reading / review

**Situation:** I'm reading code and notice two classes are suspiciously similar.

**Pattern recognition moment:** I realize this "two similar things → extract the
common structure" pattern is the same operation I'd do with skills, docs,
configs — anything with duplication.

**Action:** Check if the analogous duplication exists in the adjacent substrate
(skills, docs, configs).

**But:** This is a stretch — when reading code, I'm focused on code. Would I
really interrupt to check skills for the same pattern?

### Trigger 3: During skill updates (most promising?)

**Situation:** I'm adding a rule to a skill file. I notice the rule looks
remarkably like a code pattern I already know.

**Pattern recognition moment:** "Wait — this rule is literally dependency
injection, applied to skills." Or: "This 'trace the execution path' debugging
procedure is the same thing as stepping through a call stack."

**Action:** Import the battle-tested procedures from the well-known domain.
Instead of inventing a new skill rule from scratch, borrow the
checklist/procedure from the established domain (e.g., code refactoring → skill
refactoring).

### Trigger 4: During design or planning

**Situation:** I'm designing a new feature, system, or process.

**Pattern recognition moment:** The architecture I need mirrors a pattern from a
different domain that I already know well.

**Action:** Reuse the design and its known pitfalls. "This is the observer
pattern applied to config changes" → import the known failure modes (memory
leaks, ordering issues).

### Trigger 5: During learning / reading new concepts

**Situation:** I encounter a new concept, framework, or technique.

**Pattern recognition moment:** "This is the same structure as X, just in a
different domain."

**Action:** Transfer everything I know about X — shortcuts, pitfalls, optimal
strategies — to the new concept.

## Synthesis: what's the real trigger?

Looking across all these scenarios, the trigger is not one specific workflow
phase. It's a **cognitive event** that can happen at any time:

> **"This reminds me of something else in a different domain."**

The question is: can we operationalize a cognitive event as a skill trigger?

### Option A: Passive — always-on principle

Like `principles`, it's not a workflow step but a background check. We state the
principle and examples, and rely on me to notice.

**Pro:** Matches the actual trigger (spontaneous recognition). **Con:** May not
reliably fire — it's hard to trigger on "vibes."

### Option B: Active — checkpoint in specific workflows

Add an explicit "check for cross-level patterns" step to workflows where it's
most valuable:

-   Self-improvement retrospective (most natural home)
-   Skill updates (when adding a rule, ask "is this an instance of a known
    pattern from another domain?")
-   Design phase (when designing, ask "does this architecture mirror a known
    pattern?")

**Pro:** Reliably fires at known moments. **Con:** May feel forced at some
checkpoints; clutters workflows.

### Option C: Hybrid — principle + targeted checkpoints

State the general principle in its own skill file. Then add lightweight
cross-references from the workflows where it's most likely to fire:

-   `self-improvement`: "Check for cross-level patterns (see
    `cross-level-patterns`)"
-   `update-skills`: "Before writing a new rule, check if it's an instance of a
    pattern from code/design/process (see `cross-level-patterns`)"

**Pro:** Principle exists as reference; targeted reminders ensure it fires.
**Con:** More surface area to maintain.
