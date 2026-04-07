---
name: doc-style-guide
description: >
  Documentation style rules for markdown documentation pages.
  Read when writing or editing markdown documentation files.
---

# Documentation Style Guide

## Content

-   Do not add `## Feedback / Help` sections on individual doc pages. This
    section belongs only on the top-level `index.md`.
-   Think about navigation. When relevant:
    -   Update the table of content.
    -   Add cross-links to relevant questions.

## Code examples

-   Code examples implicitly teach patterns beyond their technical content (file
    structure, naming, what belongs together). Separate distinct concerns (e.g.
    model definition vs. config usage) into separate code blocks so the example
    doesn't accidentally recommend bad practices.
-   When documenting a wrapper or adapter, proactively address interactions with
    adjacent systems users are likely already using. Omitting these leads to
    reviewer/user questions that should have been answered in the docs.
