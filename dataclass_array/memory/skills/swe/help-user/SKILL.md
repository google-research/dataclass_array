---
name: help-user
description: >
  Workflow for responding to external user questions in chat channels.
  Applies fix-the-process: answer the question AND fix what made it
  necessary.
---

Every user question is a bug report. The question itself signals a gap — missing
docs, bad error message, or missing feature. Fix the gap first, then answer. See
`fix-the-process` for the underlying principle.

## Workflow

When the user points to a question from an external user (e.g. a chat message):

### 1. Read the full thread

Using `gchat` skill:

Use `read_thread` (not just `get_message`) to get the full conversation context.
Other replies may already answer the question or add important details.

### 2. Fix the meta-problem

Ask: *why couldn't they find the answer themselves?*

Typical root causes and fixes:

-   Missing documentation → create/update docs.
-   Bad error message → improve the error.
-   Missing feature → implement it or file a bug.

Do the fix, create a CL if needed.

### 3. Draft the reply

Follow `write-message` for format and rules.

### 4. Send

After the user approves the draft, send the message via the gchat CLI.
