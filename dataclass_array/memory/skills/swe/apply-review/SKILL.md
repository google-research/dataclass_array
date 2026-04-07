---
name: apply-review
description: >
  Apply reviewer comments from a Critique CL. Read comments, fix code, reply,
  and iterate until all comments are resolved.
---

## Workflow

When asked to apply review comments from a CL (e.g. "review cl/XXXXXX"):

### 1. Read comments

Use `fetch_changelist` (with `include_reviewer_comments=true` and
`include_code_changes=true`) to get:

*   The full diff (to understand current code changes).
*   All unresolved reviewer comments.

### 2. Apply fixes

For each unresolved comment:

1.  Understand what the reviewer is asking.
2.  Make the code change. If the comment is ambiguous, ask the user for
    clarification via `notify_user` before proceeding.
3.  **Apply `fix-the-process`**: each reviewer comment is a signal.
4.  If a comment reveals a limitation worth fixing but out of scope, track it
    via `write-todo`.
5.  Build and test to verify the fix.

### 3. Amend changes to the CL

After all fixes are applied and verified, run `hg amend` to fold the changes
into the current CL.

### 4. Reply to comments

Use the **critique skill** to reply to each comment:

*   **Trivial / mechanical changes** (e.g. "rename X", "use auto()", "remove
    quotes"): reply with `"Done"` and `--resolve`.
*   **Non-trivial changes** where the fix differs from what was suggested: reply
    with a brief explanation of what was done and why, and `--resolve`.
*   **Disagreements or questions**: reply with reasoning but do **not** resolve.
    Flag to the user via `notify_user`.
*   Always starts your reply by `🤖` emoji.
*   If the comments does not originate from you (the cl author), validate your
    solution before answering.
*   If the comment spawned a TODO, link the TODO file in the reply.

To reply, first fetch comment IDs:

```bash
critique_tool --cl=<CL_NUMBER> --comments
```

Then reply to each:

```bash
critique_tool --cl=<CL_NUMBER> --reply=<COMMENT_ID> --message="🤖 Done" --resolve
```

### 5. Iterate

If the reviewer leaves new comments after the next round, repeat from step 1.

## Rules

*   Always build and run relevant tests **before** replying to comments.
*   Do not silently skip comments. Every unresolved comment must be addressed.
*   Group related fixes together but reply to each comment individually.
