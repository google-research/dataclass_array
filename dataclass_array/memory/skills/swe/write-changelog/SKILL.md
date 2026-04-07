---
name: write-changelog
description: >
  Write an exhaustive changelog from version-control history. Use when preparing
  a release or updating CHANGELOG.md.
---

## Core principle

A changelog is only as good as its completeness. Every user-visible change must
appear. Missing entries erode trust in the release notes.

## Procedure

### Phase 1: Establish the boundary

1.  Find the **exact CL (or commit) of the previous release**. Don't guess —
    query for the tag, version bump CL, or release marker explicitly.
2.  Verify the boundary independently: check the date, description, and modified
    files match what you expect for a release CL.
3.  Note the boundary CL number and date for use in Phase 2.
4.  Count the number of cls for use in Phase 2

### Phase 2: Extract & classify raw data

1.  Query *all* CLs in the range `[boundary_CL, now]` for the relevant directory
    (output directly in the user `experimental/`). Use `g4 changes -l` (long
    format) to get full descriptions.
2.  For each CL, classify it based on the **full description**:
    -   **Changelog-worthy** — new features, bug fixes, API changes,
        deprecations, performance improvements, documentation.
    -   **Not changelog-worthy** — rollbacks that were re-landed, pure
        visibility changes, automated formatting, BUILD migrations,...
3.  If the feature is not clear from the description, look at the actual files
    to get a better understanding. This is most important for big changes.
4.  Write concise, user-facing descriptions — not CL titles. Explain *what
    changed for the user*, not the implementation detail. Each entry should be
    understandable without reading the CL — if a reader can't tell what the
    feature does from the one-liner, look at the cl to make the description
    clear.
5.  Each new feature should be documented separately. If a new feature was
    implemented in multiple cls (e.g. a new module), it should only be
    documented once.

If the number of cl is large, **follow `batch-processing` very thoughtfully**
for the chunking strategy**.

### Phase 3: Write the changelog

1.  **Read the existing changelog** for formatting conventions.
2.  Use the **Formatting conventions** below.
3.  Group entries by module/component using nested bullet points. The
4.  Order within each group using the **Tag ordering** below.
5.  If the release has many features, highlight a few important ones at the top.
    You can include very short minimal code snippet for major feature, but do
    not overdo it (no more than 1-2 max).

### Phase 4: Cross-reference (mandatory)

Before presenting the changelog:

1.  Go through **every** changelog-worthy CL from Phase 2.
2.  Verify each one has a corresponding entry in the written changelog.
3.  Add any missing entries.
4.  Record how many were missing and why — this is your quality signal.

### Phase 5: Present with audit trail

When presenting to the user:

-   State the exact counts: total CLs, PUBLIC, changelog-worthy, entries
    written.
-   Offer the full CL listing as a separate artifact for verification.
-   Flag any CLs where the classification was uncertain.

## Formatting conventions

-   Group by module, nest individual changes with a tag and symbol:

    ```markdown
    * `module` / category:
      * [Tag] `symbol`: Short description
      ...
    ```

-   New modules should be documented as:

    ```markdown
    * [New] `module`: Short description
    ```

-   `module` correspond to the public API namespace used by the end-user (e.g.
    `np`, `np.linalg`,...)

-   **Module ordering**: New modules (entirely new top-level additions) appear
    first with the `[New]` tag, before changes to existing modules.

-   **Tag ordering** within each module: `[New]`, `[Extended]`, `[Changed]`,
    `[Removed]`, `[Breaking]`, `[Fix]`.

    | Tag          | Use for                                                  |
    | ------------ | -------------------------------------------------------- |
    | `[New]`      | Big features and new public symbols (classes, functions, |
    :              : modules). Not for small additions to existing APIs.      :
    | `[Extended]` | Extending existing features: new parameters, options, or |
    :              : supported types on an existing symbol.                   :
    | `[Changed]`  | Behavior changes, renames, or internal reworks of        |
    :              : existing features.                                       :
    | `[Removed]`  | Deprecated or deleted public symbols.                    |
    | `[Breaking]` | Changes that require users to update their code.         |
    | `[Fix]`      | Bug fixes.                                               |

-   Include version number and date in the header: `## [X.Y.Z] - YYYY-MM-DD`.

-   Update version-comparison links at the bottom of the file.

## Anti-patterns

-   ❌ Starting to write before verifying the boundary CL.
-   ❌ Declaring "~N changes covered" without cross-referencing.
-   ❌ Using CL titles verbatim — they're written for reviewers, not users.
-   ❌ Writing entries that only make sense if you already know the feature —
    every entry must be self-explanatory to a library user.
-   ❌ Skipping non-PUBLIC CLs entirely — some may contain user-visible fixes.
-   ❌ Pre-filtering CLs by directory path — a CL may touch both project-specific
    and core library files. Classify the full CL, not its path.
-   ❌ Classifying CLs from truncated `g4 changes` output — the default format
    shows ~30 chars per title. Always use `-l` for full descriptions.
