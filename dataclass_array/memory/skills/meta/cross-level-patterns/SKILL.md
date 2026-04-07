---
name: cross-level-patterns
description: >
  Recognize when two things at different abstraction levels or domains are
  structurally the same. Extract the common pattern and factorize it.
---

## The principle

When something feels structurally familiar — "I've seen this shape before, in a
different domain" — that's a cross-level pattern. The same structure often
appears across different substrates: code, skills, docs, processes, designs.

This is distinct from `multi-level-thinking` (which walks one concept up through
abstraction levels). Cross-level recognition works **diagonally** — it notices
two things at *different* levels are the same, and extracts the common structure.

## How to apply

1.  **Recognize.** Notice the structural similarity. "This feels like X from
    domain Y."
2.  **Validate.** The analogy must share **actionable structure**, not just
    vocabulary. Ask: "is there a common procedure, failure mode, or design
    principle that governs both instances?" If only the names map, it's a false
    analogy.
3.  **Extract.** Factor out the common pattern — name it, write it once, and let
    both instances reference the shared structure.

## Examples

| Pattern             | Domain A                  | Domain B                   |
| ------------------- | ------------------------- | -------------------------- |
| **Deduplication**   | Extract shared function   | Extract shared principle   |
:                     : from duplicate code       : from duplicate skill rules :
| **Layered           | Code: utilities → domain  | Skills: principles →       |
: architecture**      : → API                     : workflows →                :
:                     :                           : situation-specific rules   :
| **Dead code         | Unreachable code branch → | Skill rule never on        |
: elimination**       : delete                    : execution path → move or   :
:                     :                           : delete                     :
| **Test-first**      | Write the test before the | Define correct behavior    |
:                     : implementation (TDD)      : before fixing a skill      :
:                     :                           : (counterfactual test)      :
| **Trace execution** | Debug by stepping through | Diagnose a skill failure   |
:                     : the call stack            : by tracing the skill graph :
| **Dependency        | List imports at top of    | List skills at top of      |
: declaration**       : file                      : implementation plan        :
| **Single            | One function does one     | One skill covers one       |
: responsibility**    : thing                     : trigger                    :
