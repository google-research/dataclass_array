---
name: send-review
description: >
  Review someone else's CL and post comments via Critique. Use when asked to
  review a CL or send review comments.
---

## When to use

When reviewing someone else's CL (not applying received reviews — see
`apply-review` for that).

## Workflow

1.  Read relevant skills (`api-design`, `py-style-guide`, `test-design`, etc.)
    before looking at the CL.
2.  Fetch the CL with `fetch_changelist` (diffs + comments).
3.  Analyze against the skills.
4.  Draft comments and present to user for validation **before** posting via the
    `critique` skill.

## Comment style

*   Prefix every comment with `🤖`.
*   Add a type tag after the emoji: `[API]`, `[Bug]`, `[Test]`, `[Formatting]`,
    or `(nit)`.
*   **Questions over commands.** Point out the issue and ask if the alternative
    was considered — don't tell the author what to do. Bad: *"Mutates in-place
    and returns — pick one."* Good: *"This both mutates and returns. Should it
    copy the config first?"*
*   **Compress, don't truncate.** Context is implied by placement, so don't
    repeat what the reader can see. But comments can be longer when the issue
    warrants it — just no redundancy.
*   **Don't explain twice.** State the issue once. If the fix is obvious, don't
    spell it out.

## Review priorities

Prioritize **architecture and caller-side experience** over internal
consistency. The most valuable review comments are about whether the
abstractions are right, whether the API makes sense from the caller's
perspective, and practical concerns (backward compatibility, migration paths).
Style nits and test gaps matter, but they are secondary.

## Related

Style guides:

-   `api-design` — architecture rules to check during review
*   `swe-style-guide` — General coding practices to follow.
*   `py-style-guide` — code quality rules to check during review
*   `doc-style-guide` — documentation quality rules to check during review. Also
    apply to code documentation.
*   `test-design` — test quality rules to check during review
