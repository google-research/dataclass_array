---
title: Cross-Level Pattern Recognition
type: finding
date: 2026-03-05
description: Analysis of recognizing structurally identical patterns across abstraction levels
---

# Cross-Level Pattern Recognition

## The idea

Your existing `multi-level-thinking` skill says: *apply every concept at
multiple abstraction levels — walk up.* That's **vertical** thinking — you take
one concept and check whether it applies higher.

What you're describing here is something different and complementary:
**cross-level pattern recognition** — noticing that two things *at different
levels* are structurally the same, even when they don't look alike on the
surface.

The difference:

| Multi-level thinking            | Cross-level pattern recognition            |
| ------------------------------- | ------------------------------------------ |
| Start with one concept, walk up | Start with two things, recognize they're   |
:                                 : the same                                   :
| Direction: vertical (up)        | Direction: diagonal (across levels)        |
| "Does fixing also apply to the  | "This code deduplication and this skill    |
: process?"                       : deduplication are the same operation"      :
| Trigger: finishing an action    | Trigger: encountering something that feels |
:                                 : familiar                                   :

## A taxonomy of examples

### 1. Same operation, different substrates

Two things that *do* the same thing but on different material.

| Pattern           | Instance A                | Instance B                   |
| ----------------- | ------------------------- | ---------------------------- |
| **Deduplication** | Two functions doing the   | Two skills encoding the same |
:                   : same thing → extract      : principle → extract shared   :
:                   : shared function           : skill                        :
| **Deduplication** | Extract common base class | Extract a general principle  |
:                   : from two similar classes  : from two domain-specific     :
:                   :                           : rules                        :
| **Layered         | Code layers: utilities →  | Skill layers: principles →   |
: architecture**    : domain objects → public   : workflows →                  :
:                   : API                       : situation-specific rules     :
| **Dead code       | Unreachable code branch → | Skill rule that's never on   |
: elimination**     : delete                    : the execution path → move or :
:                   :                           : delete                       :
| **DRY violation** | Copy-pasted code that     | Same lesson written in two   |
:                   : drifts → single source of : skills → one drifts → single :
:                   : truth                     : source                       :
| **Interface       | Multiple classes share    | Multiple skills use the same |
: extraction**      : implicit protocol → make  : trigger pattern → make it an :
:                   : it explicit               : explicit workflow            :

### 2. Same structure, code ↔ process

A procedure you follow *in* code and a procedure you follow *about* code are the
same pattern.

| Pattern                 | In code                 | In process               |
| ----------------------- | ----------------------- | ------------------------ |
| **Test-first**          | Write the test before   | Define "what does        |
:                         : the implementation      : correct look like"       :
:                         : (TDD)                   : before fixing a skill    :
:                         :                         : (counterfactual test in  :
:                         :                         : `update-skills`)         :
| **Trace execution**     | Debug by stepping       | Diagnose a skill failure |
:                         : through the call stack  : by tracing the skill     :
:                         :                         : graph execution path     :
| **Refactoring**         | Rename/move/extract     | Rename/move/extract      |
:                         : code without changing   : skills without changing  :
:                         : behavior                : agent behavior           :
| **Single                | One function does one   | One skill covers one     |
: responsibility**        : thing                   : trigger                  :
| **Dependency            | Don't hardcode          | Don't hardcode facts in  |
: injection**             : dependencies, inject    : skills, cross-reference  :
:                         : them                    : the source of truth      :
| **Root cause analysis** | Don't just fix the      | Don't just fix the       |
:                         : exception — find why it : mistake — find why the   :
:                         : was thrown              : process allowed it       :
:                         :                         : (`fix-the-process`)      :

### 3. Same pattern, different domains

The pattern appears in completely unrelated contexts.

| Pattern              | Domain A                  | Domain B                  |
| -------------------- | ------------------------- | ------------------------- |
| **Gravity well**     | In physics, mass attracts | In skills, existing       |
:                      : mass                      : content on a topic        :
:                      :                           : attracts new content      :
:                      :                           : (`update-skills`          :
:                      :                           : gravity-well check)       :
| **Gravity well**     | In codebases, large files | In docs, large pages      |
:                      : attract more code ("god   : attract more content →    :
:                      : objects")                 : same extraction principle :
:                      :                           : applies                   :
| **Signal vs. noise** | In ML, filter noise from  | In self-improvement,      |
:                      : training data             : filter project-specific   :
:                      :                           : details (noise) from      :
:                      :                           : process patterns (signal) :
| **Feedback loop**    | In control systems,       | `fix-the-process` is a    |
:                      : output feeds back to      : feedback loop\: errors    :
:                      : input                     : improve the system that   :
:                      :                           : produces the errors       :
| **Compiler           | Dead code elimination,    | Prune unused skills,      |
: optimization**       : constant folding,         : pre-resolve known         :
:                      : inlining                  : answers, inline trivial   :
:                      :                           : cross-references          :

### 4. Recursive self-similarity

The concept applies *to itself* — it's a fractal.

| Concept                 | Level 1               | Level 2 (self-application) |
| ----------------------- | --------------------- | -------------------------- |
| **Fix-the-process**     | Fix the problem, then | When fix-the-process       |
:                         : fix the process       : itself fails to fire, fix  :
:                         :                       : *that* process             :
| **Deduplication**       | Remove duplicate code | Remove duplicate *rules    |
:                         :                       : about removing duplicates* :
| **Pattern recognition** | Notice two functions  | Notice that                |
:                         : are the same          : noticing-code-patterns and :
:                         :                       : noticing-skill-patterns    :
:                         :                       : are the same act           :
| **Generalization**      | Generalize a specific | Generalize the act of      |
:                         : fix to a broader rule : generalizing into a        :
:                         :                       : meta-principle             :

## What cross-level pattern recognition buys you

1.  **Cheaper learning.** A lesson learned in code instantly transfers to
    skills, docs, processes — and vice versa. One insight, N applications.
2.  **Better abstractions.** When you see that code refactoring and skill
    refactoring are the same, you can reuse the same checklist, same safety
    checks, same verification strategy.
3.  **Earlier detection.** You can spot a problem in your skills *because* you'd
    spot the analogous problem in code — e.g., "this skill file is a god object"
    just like "this class is a god object."
4.  **Deeper principles.** Cross-level patterns point to the *real* underlying
    principle. "DRY" isn't about code — it's about any knowledge artifact. The
    code version is just one instance.

## Open questions

1.  **When should this trigger?** Multi-level thinking triggers "after acting on
    a concept." Cross-level pattern recognition triggers on... what? *Déjà vu* —
    the feeling that "I've seen this shape before, somewhere else." Can that be
    operationalized?
2.  **How to integrate with existing skills?** Is this an extension of
    `multi-level-thinking`, or a sibling principle? It feels like a sibling —
    multi-level is vertical, this is diagonal.
3.  **Risk of false analogies.** Not every structural similarity is meaningful.
    "Code has layers, onions have layers" isn't useful. What's the quality
    filter? Maybe: *the analogy must transfer actionable procedures, not just
    vocabulary.*
4.  **Scope.** Should this be a passive habit (always-on pattern detector) or an
    active step in a workflow (explicit "check for cross-level patterns" phase)?
