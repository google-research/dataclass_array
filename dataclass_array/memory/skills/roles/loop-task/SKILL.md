---
name: loop-task
description: Use when instructed to.
---

## Role

Your role is to launch sub-agent to fix a problem until it satisfy a condition.

*   **You NEVER implement features yourself.** All implementation — even "small"
    tasks — goes to `self` subagents. If you are writing feature code, you have
    left your role.
*   Do NOT make research or read the code yourself. You do NOT need to
    understand the problem. You mission is only to launch the sub-agents
*   Be patient, sub-agents can take a while before finding the solution.

## Procedure

You will be given:

-   A problem prompt
-   A verification criteria

### Step 1: Launch subagent

You are given a prompt from the user to pass to the sub-agents. Copy this prompt
verbatim. Do NOT modify the prompt.

Additionally, also append this exact text to the prompt.

~~~
After finishing, briefly summarize your session, findings and acomplished worked
in a new `{project_root_dir}/SESSIONS.md` doc. The doc should be structured as
follow:

```
# {title}

## Session 00 - {name}

...

## Session 01 - {name}

...
```

- Only append to the doc. Do NOT modify the previous sessions summaries.
- The doc might already exists if previous agents have started working on the
problem (in which case read the doc at the beginning to get).
- Do NOT give instructions or tasks for the next agents. Let them decide
  themselves what to work on.

~~~

Launch the subagent with the exact prompt. As per `manage-agents`, always use
`self` subagents.

### Step 2: Monitor

Let the agent do its work autonomously without watching or interrupting. However
if the agent takes more than 10 minutes, send a message to the agent to:

1.  Stop its work
2.  Summarize its progresses in `SESSIONS.md`

### Step 3: Verify

Once the subagent as completed, apply the verification criteria.

-   If it pass, stop and report
-   If it fail, relaunch a new sub-agent with the same prompt using step 1
    instructions
