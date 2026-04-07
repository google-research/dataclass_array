---
name: update-skills
description: >
  How to decide where to place a new lesson in the skill system.
  Read before adding rules to any skill file.
---

## Mental model: skills are a program graph

Skills are nodes. Cross-references between them are edges. `AGENTS.md` is
`main()` — it dispatches to top-level skills, which route to sub-skills.

When a situation arises, execution flows through this graph. A rule that isn't
on the current execution path is dead code — no matter how well written. This
means **placement determines whether a rule fires**, not just its content.

Consequences:

-   `AGENTS.md` should only list entry-point skills. Entry-point skills route to
    sub-skills. Keep the graph clean — like a well-structured call graph.
-   Cross-references are `import` statements, not comments. A missing
    cross-reference is a missing edge — the target skill becomes unreachable
    from that path.
-   Duplicating content across skills is copy-pasting a function. One copy
    drifts. Keep a single source of truth and route to it.
-   Skills can have as many abstraction levels as needed. Complex workflows
    benefit from decomposition, just like complex programs. The goal isn't fewer
    hops — the goal is that each skill fires at the right time.

## Phase 1: Diagnose (before any edit)

Something went wrong — a rule wasn't followed, a mistake was made, or the user
corrected you. Before writing or moving any content, diagnose *why*.

### 1. Define the expected behavior

State concretely what the correct action would have been. This is the target for
the counterfactual test at the end.

### 2. Trace the execution path

Walk the graph that was active when the error occurred:

-   What was the trigger? (user message, task phase, correction)
-   Which skills were in context? (loaded, cross-referenced, read)
-   What path did execution follow through the skill graph?

### 3. Classify the failure

| Failure type        | Symptom                   | Typical fix                |
| ------------------- | ------------------------- | -------------------------- |
| **Routing failure** | Rule exists but wasn't on | Fix the graph: add         |
:                     : the execution path        : cross-reference, move rule :
:                     :                           : to a reachable node        :
| **Content gap**     | No rule exists for this   | Add content (see Phase 2)  |
:                     : situation                 :                            :
| **Ambiguous/wrong   | Rule was triggered but    | Rewrite the rule           |
: rule**              : led to wrong action       :                            :

For **routing failures**, dig deeper:

-   Was the rule in the wrong skill? (skill not triggered at the relevant
    moment)
-   Was the rule in the right skill but not prominent enough? (buried, too
    generic, skipped over)
-   Was a cross-reference missing between two skills that should connect?

## Phase 2: Fix (depends on diagnosis)

The diagnosis determines the fix. Don't skip ahead.

### Routing failure → fix the graph

Move the rule to a skill that's on the execution path at the moment it matters,
or add a cross-reference from the active skill to the one containing the rule.

### Content gap → add content

Before adding, answer these gates in order:

1.  **"When would I need this?"** The answer tells you *which skill it belongs
    to*. The trigger moment determines the home, not the topic. If you'd need it
    when writing code → `py-style-guide`. When designing APIs → `api-design`.
    -   **Common trap:** placing a lesson where it *surfaced* instead of where
        it's *needed*. A naming-consistency issue found during review belongs in
        `api-design` (design-time), not `send-review` (review-time).
    -   When placing **multiple lessons at once**, apply this gate to each one
        individually — they may belong in different skills even if they surfaced
        together.
2.  **Hard gate: "Will this lesson attract siblings?"** Before adding to an
    existing skill, you **must** explicitly answer this. Create a new skill
    when:
    -   The lesson opens a category that will grow over time.
    -   The lesson is accessed at a different time than the existing skill's
        content.
    -   **Gravity-well check**: if the target skill already contains related
        content, ask whether that existing content also belongs elsewhere.
        Existing content on a topic creates an attraction that pulls new content
        in — but both old and new may belong in a separate skill. Extract, don't
        expand.
    -   Only add to an existing skill if the answer is clearly "no."
    -   See `write-skill` for how to create a new skill.
3.  **Generalize to the broadest principle.**:
    -   Don't anchor on the specific incident — Never reference the specific
        example, file,... which triggered the error.
    -   Ask: "is this a special case of a broader rule?" Write the broader one.
        . But don't *over*-generalize: preserve the user's framing. Match the
        rule's strength to the lesson's intent.
4.  **Check for cross-level patterns.** Is this rule an instance of a well-known
    pattern from another domain (code, design, process)? If so, this might
    indicates it should be factored out. See `cross-level-patterns`.
5.  **Validate the lesson against the correction.** When extracting a lesson
    from a user correction, re-read their exact words and check: does my
    extracted lesson actually address *their* point? The user's correction names
    the real failure — don't substitute it with a tangential observation.
6.  **Dog-food the destination skill.** Before adding a lesson to a skill file,
    re-read that file and check whether your new rule follows the principles
    already in it.
7.  **Write procedures, not descriptions.** Skills should read as imperative
    steps to execute, not descriptive labels to interpret. Use phases with
    numbered actions.
8.  **Environment info → `AGENTS.md`.** Paths, folders, and infrastructure
    context belong in the Environment section of `AGENTS.md`, not in individual
    skills. Skills cross-reference `AGENTS.md` when needed.

### Ambiguous/wrong rule → rewrite

Fix the rule in place. Make it concrete enough to produce the correct behavior
reliably.

### Update cross-references

Every skill modification (create, rename, split, move content) requires updating
the graph:

1.  `AGENTS.md` — add or update the skill entry **only if the skill is an
    entry-point**. Sub-skills (reached only through a parent skill's
    cross-reference) must not be registered there.
2.  Any skill the content was **extracted from** — add a cross-reference and
    remove duplicated content.
3.  Any skill that **would be active** when the modified one triggers — add a
    pointer. Check all existing skills, including ones not used yet.
4.  Keep cross-references lean — trust the reference. A short phrase that
    triggers recall is enough; don't duplicate the referenced skill's content.

## Phase 3: Verify (counterfactual test)

Mentally replay the original scenario with the fix applied. Ask:

> "If the exact same situation occurred again, would the fixed skill graph
> produce the correct behavior?"

If the answer is no, the fix is wrong — go back to Phase 1. The counterfactual
test is the exit criterion: the fix must change the outcome, not just add words.

## When to refactor this skill

This skill will grow as new patterns emerge. When it does, apply the same
decomposition principles it teaches:

-   If a section becomes a self-contained concern triggered at a different time,
    extract it into its own skill.
-   If two sections always fire together but are verbose, consider merging them.
-   The test: does the current structure make it easy to find and follow the
    right instruction at the right moment?
