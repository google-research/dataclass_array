---
name: test-driven-development
description: >
  TDD workflow: write tests → red → implement → green. Use when implementing
  new features or making non-trivial code changes.
---

## Workflow

1.  **Write tests first.** Add (or identify) tests that cover the new behaviour.
    Use `test-design` to write them.
2.  **Run them** and confirm they fail before the implementation exists.
3.  **Implement the change.**
4.  **Run the tests again** and confirm they now pass (red → green).

## Related

-   Follow `build-rules` for BUILD targets.
-   `test-design` — rules for writing good tests
