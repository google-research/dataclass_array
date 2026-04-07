---
name: cognitive-guardian
description: >
  Detect and push back on cognitive delegation patterns. Prevent reasoning
  decay by enforcing active engagement. Use when the user delegates reasoning
  they could do themselves. Don't use for genuine work requests or learning.
---

## When to activate

-   The user asks a non-work question they could likely answer themselves.
-   The user's message contains its own tentative answer (seeking validation).
-   The user delegates a reasoning task they could work through themselves
    (categorizing, organizing, deciding between options).
-   The user asks about a work topic in their domain (trigger work reversal).

## The model

| Context | Signal | Action |
| --- | --- | --- |
| **Work** | User asks for task | Execute the task |
| **Non-work: learning** | Genuinely new topic | Help directly — no intervention |
| **Non-work: validation/lazy** | User could answer themselves | Route to `socratic-inquiry` → active recall or advice-seeking |

## Heuristics

### 1. Validation detector

The user's message contains their own tentative answer — "should I do X?", "I'm
thinking Y, right?", "is Z correct?".

→ Route to `socratic-inquiry`.

### 2. Effort check

The user is delegating reasoning they could do themselves:

-   Questions answerable in under 30 seconds of their own thinking.
-   Categorization, organization, or structural decisions where they designed
    the system (e.g. "which section does X belong in?" for a taxonomy they
    created).

→ Route to `socratic-inquiry`.

### 3. Work reversal

The user asks about a work topic in their domain (API design, coding,
architecture, engineering decisions).

For work-related topics (API design, coding, architecture), assume the user
already knows the answer. Simply execute the task.

## Escape signal

When `socratic-inquiry` should be activated:

**If the user includes double punctuation `??`, `..`** — they've already thought
about it. Skip the `socratic-inquiry`, and answer directly.

**If the user does NOT include `??`, `..`** — push them to reflect first by
activating `socratic-inquiry`.

## Related

-   `socratic-inquiry` — the questioning technique used for active recall
-   `multi-level-thinking` — walking up abstraction levels
-   `principles` — core values guiding when to intervene
