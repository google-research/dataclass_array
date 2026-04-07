---
name: test-design
description: >
  High-level rules for good tests. Use when writing or reviewing tests.
---

## Core Rules

-   Use `test-driven-development` — TDD workflow (write test → red → implement →
    green)
-   Include one integration test that exercises the feature through its actual
    entry point, not just the isolated utility function.
-   **Red-green testing for bug fixes and features:** When fixing a bug, first
    write (or identify) a test that reproduces the failure, verify it fails,
    then apply the fix and confirm the test passes. Never add a test after the
    fix without first verifying it would have failed before.
-   Include **edge cases and adversarial inputs** from the start — e.g. special
    characters in multiple positions, mixed concerns (same character meaning
    different things in different contexts), quote/escape variations, and empty
    inputs.
-   **Cover all affected code paths:** When a fix changes shared logic (e.g. a
    constraint or type definition used by multiple paths), test every path — not
    just the one that triggered the bug. For example, if a type accepts both
    format A and format B, test both even if only one was broken.
-   **Parametrize repetitive tests.** When multiple test functions differ only
    in inputs and expected outputs, proactively collapse them into a single
    `@pytest.mark.parametrize` test with `pytest.param(..., id=...)` entries.
    Don't wait for the user to notice the duplication.
-   **Every specified behavior must be tested.** If a behavior is documented —
    in a docstring, CL description, or design doc — it must have a corresponding
    test. When adding a new capability, add the test at the same time as the
    documentation.

## Python style

-   Use `pytest` framework.
-   Do not add `if __name__ == "__main__":` blocks.
-   Do not add module docstring.
-   Test file mirrors source file: `module.py` → `module_test.py` in the same
    folder.

## Pytest Plugins

-   To use Pytest plugins (e.g. `pytest_aiohttp`): Add the plugin dep in BUILD,
    then register it explicitly in the test file using the open-source import
    path directly:

    ```python
    pytest_plugins = ("pytest_asyncio.plugin",)
    pytestmark = pytest.mark.asyncio

    async def test_something(aiohttp_client, app):
        cli = await aiohttp_client(app)
        ...
    ```

    The `pytestmark` enables async test collection without per-test decorators.

## Related

-   `swe-style-guide` — general engineering conventions
-   `py-style-guide` — Python-specific test conventions (pytest, no `__main__`)
