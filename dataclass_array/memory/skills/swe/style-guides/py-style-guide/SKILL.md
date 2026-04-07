---
name: py-style-guide
description: >
  Code style conventions, formatting, and file structure rules.
  Read BEFORE implementation when writing code (big or small).
---

# Style Guide

## File structure

Every `.py` file follows this order:

1.  Module docstring — short, e.g. `"""Ray utils."""`
2.  Imports category, each alphabetical and separated by blank line:
    1.  Always add `from __future__ import annotations`
    2.  Stdlib imports (alphabetical)
    3.  Third-party / project imports
    4.  Project-local imports.
3.  Module-level constants / aliases
4.  Classes and functions. Publicly exposed modules should appear first.
    Internal functions should be moved at the end.

-   **No unnecessary lazy imports.** Do not preemptively use function-level
    imports to avoid circular deps — verify the circular dependency actually
    exists first. Top-level imports are always preferred.

## Docstrings

-   Include usage examples with ` ```python` code blocks for key APIs.

## Error handling

-   A bad error message is a bug. Errors should be clear and self-explanatory.
-   Use `epy.reraise(e, prefix=...)` to add context to exceptions to help user
    debug.

## Enums

-   Always use `enum.auto()` for enum values, not string literals or manual
    integers.
-   Prefer `enum.StrEnum` for configuration options. User-facing functions
    should accept `MyEnum | str` and `str` get normalized internally.
-   Document each enum member in an `Attributes:` section of the class
    docstring:

    ```python
    class MyEnum(enum.StrEnum):
      """One-line summary.

      Attributes:
        VALUE_A: Description of VALUE_A.
        VALUE_B: Description of VALUE_B.
      """

      VALUE_A = enum.auto()
      VALUE_B = enum.auto()
    ```

## Type annotations

-   Use bare union syntax (`int | str`) instead of string-quoted annotations
    (`'int | str'`). This project runs Python 3.12+.
-   **Readability over pytype exactitude.** If the code is correct but pytype
    complains (e.g. `Unpack` typing issues in third-party libs), suppress with
    `# pytype: disable=xxx`. Do not distort code structure to satisfy pytype.

## Comments

-   **Never remove existing comments** during a refactoring or feature change,
    even if they seem obvious. Comments are there for a reason. Only remove a
    comment if it is factually wrong or the code it describes has been deleted.

## Misc

-   Use `del arg # Unused` to indicates unused function argument.
-   `# TODO(username):` format for TODOs.

## Related

-   `test-design` — Python style guide for test. Read before writing tests.
