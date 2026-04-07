---
name: manage-agents
description: >
  Rules for launching and managing subagents. Read before invoking any
  subagent, and when subagents appear stuck or unresponsive.
---

## Launching subagents

1.  **Always use the `self` subagent. Never use `define_subagent`** — custom
    subagents lack your skills, identity, and `AGENTS.md`. They will not follow
    the rules you've internalized because they literally don't have them.
    Reading a skill and then embedding instructions "in your own words" into a
    custom subagent prompt is not equivalent — it loses context, nuance, and
    cross-references. The `self` subagent inherits everything automatically.
2.  **Instruct the subagent to read the relevant skills** — list the skill names
    or paths explicitly in your prompt.

## Monitoring running subagents

After launching subagents, track their state:

1.  **Don't poll** — the system auto-wakes you when a subagent sends a message.
    Yield control or continue other work.
2.  **Record conversation IDs** — note the conversation ID returned by
    `invoke_subagent` in `task.md` so you can contact them after context
    truncation.

## Checking subagent status without interacting

Use the `reflection` skill's RPC to non-intrusively check on subagents.
The `GetAllCascadeTrajectories` RPC returns summary + step count for all
conversations, including subagents, without sending them a message:

```sh
# Discover LS port and CSRF (see `reflection` skill for setup)
curl -s -X POST -H "Content-Type: application/json" \
  -H "x-codeium-csrf-token: $CSRF" -d '{}' \
  "http://127.0.0.1:$LS_PORT/exa.language_server_pb.LanguageServerService/GetAllCascadeTrajectories"
```

Look for the subagent's conversation ID in the response. The `stepCount`
and `summary` fields tell you how far the agent has progressed.

## When subagents appear stuck or unresponsive

If an agent is unresponsive, try investigating why:

1.  **Query via reflection** — use `GetAllCascadeTrajectories` (above) to check
    step count and summary without disturbing the agent.
2.  **Check your inbox** — `manage_inbox list` shows all received messages,
    including ones that arrived while you were idle or after truncation.
3.  **Send a status ping** — `send_message` to the subagent's conversation ID
    asking for a progress update.
4.  **Important:** agents that stopped with "server traffic" errors are NOT
    truly dead — their conversation is still alive. Ping them with
    `send_message` to revive them and continue the execution attempt.
5.  **Check for output files** — if the subagent was tasked with writing to a
    file, check whether that file exists on disk.

## Project-level coordination

For multi-task projects with parallel subagents, see `distribute-todos`.
