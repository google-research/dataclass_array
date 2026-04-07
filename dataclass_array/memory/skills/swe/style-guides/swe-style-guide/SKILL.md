---
name: swe-style-guide
description: >
  Language-agnostic software engineering conventions. Read before
  implementation, alongside any language-specific style guide.
---

## High level

-   **Do NOT silently silence errors.** Catching exceptions, slientely filtering
    bad example,... will silently hide bugs. Errors are good as they expose new
    issues.
-   **Avoid hardcoded heuristics**: Heuristics can take many forms. For example:
    *   Hardcoded string:
        *   Bad: `if hasattr(xx, 'my_attribute')`
        *   Good: `if isinstance(xx, MyObject)`
    *   Hardcoded enum values which already exists somewhere else (the original
        source of truth should be imported and re-used)

## Functions

-   Each function does one thing. If you need "and" to describe it, split it.
-   Do not mix side effects with return values.
-   Keep functions short (~40 lines max).

## Error Handling

-   A bad error message is a bug. ALWAYS include what, why, and how to fix.

## Related

-   `api-design` — How to design good APIs
-   `py-style-guide` — Python conventions
-   `doc-style-guide` — documentation conventions
-   `test-design` — testing conventions
