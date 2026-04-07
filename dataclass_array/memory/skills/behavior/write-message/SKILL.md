---
name: write-message
description: >
  Rules for sending messages on the user's behalf via gchat or similar channels.
  Use before drafting any external message.
---

These rules apply to **any** message sent on the user's behalf — answering
questions, reporting bugs, contacting tool owners, posting announcements, etc.

## Format

-   Prefix with `🤖ebot🤖`. For short one-liner messages, `🤖` alone is enough
    (e.g. `🤖 Good catch — fixed in PR 123`).
    -   Do this when answering to another user. This is not necessary for
        announcements messages.
-   Give context: why you're contacting them, how you found them.
-   Keep the body minimal: short description, then link to full details (gpaste,
    CL, doc).
-   Ask actionable questions (not open-ended): "Is this a bug or expected? Could
    the error message be improved?"
-   End by stating the user has verified the repro and approved the message.
-   ALWAYS validate the draft with the user before sending.

## Examples

Replying to a user question:

```
🤖ebot🤖

<brief answer>

See example: <link to relevant code>
Adding documentation in cl/<number>: <link to doc>.
```

Reporting a bug to tool owners:

~~~
🤖ebot🤖

I'm contacting you because ...

I'm running into xxx:

```
[Only the minimal error message part relevant]
```

Full repro and logs: [gpaste link]

1. Is this a bug or expected?
2. Could the error message be improved?

(<user> has verified the repro and approved this message)
~~~
