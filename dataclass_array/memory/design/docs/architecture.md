# Architecture

> **Prerequisite:** This document assumes you understand Antigravity, the Language
> Server, the GoB repo, and the plugin system. If not, read
> [concepts.md](concepts.md) first.

This document describes how the Agent Manager v2 components fit together. Read
this before modifying the gateway, supervisor, or routing logic.

## Overview

The system has three tiers:

```
┌─────────────────────────────────────────────────────────────────┐
│  Tier 1 — CLI (Python)                                          │
│  agent_manager.py  build / start / stop / status                   │
│  Responsibility: build, orchestrate, write registry             │
└─────────────────────────────┬───────────────────────────────────┘
                              │ spawns (via tmux)
┌─────────────────────────────▼───────────────────────────────────┐
│  Tier 2 — Gateway (Go, port 3001)                               │
│  Responsibility: route requests, supervise children             │
│                                                                  │
│  ┌─────────────────┐    ┌────────────────────────────────────┐  │
│  │  HTTP Router    │    │  Supervisor                        │  │
│  │  server.go      │    │  supervisor.go                     │  │
│  │                 │    │                                    │  │
│  │  /api/gateway/* │    │  Spawns + restarts:                │  │
│  │  /head/*        │    │    • Sidecar  (port 3000)          │  │
│  │  /<ns>/*        │    │    • Language Server (port 3002)   │  │
│  └────────┬────────┘    └────────────────────────────────────┘  │
│           │ proxy.go                                             │
└───────────┼─────────────────────────────────────────────────────┘
            │
   ┌────────┴─────────┬────────────────────────┐
   │                  │                        │
   ▼ :3000            ▼ :3002                  ▼ registry
┌──────────┐     ┌──────────────┐     ┌────────────────────┐
│  Sidecar │     │  Language    │     │  ~/.agent_manager/ │
│  (Go)    │     │  Server (C++)│     │  namespaces/*.json │
└──────────┘     └──────────────┘     └────────────────────┘
  Tier 3 —
  Backend services (managed as child processes)
```

## Component responsibilities

### CLI (`cli/`)

The CLI is the **build-time orchestrator**. It is the only component that:
- Compile Go binaries
- Creates `git worktree` directories for workspace isolation
- Runs `npm install` + `vite build` for frontend compilation
- Writes workspace JSON files to the registry directory

The CLI is **never running** during normal operation — it exits after starting the
gateway. The gateway takes over from there.

**Modules:**

File               | Purpose
------------------ | --------------------------------------------
`agent_manager.py` | Entry point; dispatches subcommands
`cli/process.py`   | tmux session management + port polling
`cli/registry.py`  | `WorkspaceEntry` dataclass + atomic JSON I/O
`cli/builder.py`   | `build`, `git worktree`, `npm`, `hg status`

### Gateway (`gateway/`)

The gateway is the **single long-running process**. It has two jobs:

**1. HTTP routing** (`server.go`, `proxy.go`)

Every request path starts with a namespace prefix:

```
/head/<rest>      →  namespace pinned as HEAD (resolved via ~/.agent_manager/head)
/<ns>/<rest>      →  namespace stack from registry
/api/gateway/*    →  internal gateway API (health, namespace list, head pointer)
/                 →  400 — absolute path leak detector (see below)
```

Within a namespace, the path `<rest>` is split:

```
/api/*    →  sidecar (port from registry entry)
/*        →  language server (port from registry entry)
```

This split exists because the sidecar owns the custom `/api/` endpoints while
the Language Server owns the Antigravity UI (static assets from `dist/` + WebSocket
for real-time agent communication). The LS serves `dist/` — the compiled bundle
that has our plugins embedded via the GoB repo patch pipeline.

**2. Child process supervision** (`supervisor.go`)

The gateway spawns the Sidecar and Language Server at startup and restarts them
if they crash, using exponential backoff (500 ms → 1 s → 2 s → … → 30 s, max 10
restarts). If a child exceeds the restart limit the gateway logs an error but
continues serving — the namespace will return 502 errors until the process is
manually restarted (via `agent_manager.py stop` + `start`).

**Files:**

| File | Responsibility |
|---|---|
| `main.go` | Flag parsing, wires supervisor + HTTP server |
| `supervisor.go` | `Supervisor` struct: spawn/restart child processes |
| `server.go` | HTTP mux, middleware, path routing logic |
| `proxy.go` | `namespaceProxy`: routes `/api/*` to sidecar, `/*` to LS |
| `registry.go` | Reads `~/.agent_manager/workspaces/*.json`, TTL cache |

### Sidecar (`sidecar/`)

The current sidecar is a **minimal stub**. It returns:
- `GET /api/health` → `{"status": "ok", "timestamp": <unix>}`
- `GET /api/todos` → `[]`
- `GET /api/conversations` → list of recorded conversations (PostgreSQL-backed)
- `POST /api/conversations` → record a new conversation
- `GET /api/templates` → `[]`

See [conversation_recording.md](conversation_recording.md) for how conversations
are intercepted in the frontend and persisted here.

The full sidecar (Notion sync, native file store, template management) was
implemented in v1 at `v1/sidecar/`. Those features will be ported to v2 as
the sidecar matures.

The sidecar is a reverse proxy necessity: the Antigravity frontend can only call
APIs on its own origin. By sitting in front of the LS, the sidecar intercepts
`/api/*` requests for our custom endpoints without modifying the LS itself.
See [concepts.md § Why is there a Sidecar?](concepts.md#why-is-there-a-sidecar)

## Registry

The **namespace registry** is the shared state between the CLI and the gateway.

- **Format**: one JSON file per namespace at `~/.agent_manager/namespaces/<name>.json`
- **Written by**: `agent_manager.py build` (auto-detects CitC workspace name)
- **Read by**: the gateway, via `registry.go`, with a 2-second TTL cache

```json
{
  "name": "feat-x",
  "path": "/home/user/.agent_manager/worktrees/feat-x",
  "sidecar_port": 3000,
  "ls_port": 3002,
  "has_go_edits": false,
  "bundle_path": "/home/user/.agent_manager/worktrees/feat-x/exa/agent_ui_toolkit/dev/dist"
}
```

`path` points to the `git worktree` root of the **GoB repo** (not the monorepo).
`bundle_path` is the compiled Antigravity frontend `dist/` inside that worktree —
the JS bundle with our plugins wired in by `wirePlugins()`.

`has_go_edits` is true if `hg status` shows changes under the sidecar directory
in the CitC workspace. When true, the workspace needs an isolated sidecar on a
different port (not yet implemented — currently all namespaces share port 3000).

The gateway does **not** need to be restarted when a new namespace is registered.
The TTL cache picks it up within 2 seconds.


## Root isolation middleware

Any request to the bare root path `/` is a bug — it means some code is making an
absolute fetch like `fetch("/api/todos")` instead of a relative one like
`fetch("api/todos")`.

The gateway returns **HTTP 400** with a diagnostic HTML page that shows:
- The exact path that leaked
- The `Referer` header (which identifies the plugin/page making the bad request)

This makes path-prefix bugs immediately visible rather than silently routing to
the wrong namespace.

## Data flow: a request to `/feat-x/api/todos`

```
1. Browser sends GET /feat-x/api/todos
2. server.go:  splitFirstSegment("/feat-x/api/todos") → ("feat-x", "/api/todos")
3. server.go:  seg == "feat-x" → look up registry
4. registry.go: read ~/.agent_manager/namespaces/feat-x.json (from cache or disk)
5. server.go:  newNamespaceProxy(entry.SidecarPort, entry.LSPort)
6. proxy.go:   r.URL.Path = "/api/todos"
7. proxy.go:   isAPIPath("/api/todos") == true → forward to sidecar :3000
8. Sidecar:    handleStubList → responds []
```

## Data flow: starting the stack

```
Developer:         $ ./agent_manager.py start
agent_manager.py:  1. Checks am-gateway tmux session doesn't exist
                   2. Assembles gateway command with --sidecar_bin and --ls_bin
                   3. Calls: tmux new-session -d -s am-gateway "<gateway cmd>"
                   4. Waits for :3001 to accept connections (up to 20s)

gateway (main.go): 1. Parses flags
                   2. Creates registry (reads ~/.agent_manager/workspaces/)
                   3. Creates HeadPointer (reads ~/.agent_manager/head)
                   4. Calls supervisor.Start(ctx)
supervisor.go:     5. (No processes started yet — lazy on first request)
gateway (main.go): 6. Registers HTTP routes
                   7. srv.ListenAndServe(":3001")
```

### HEAD pointer

The file `~/.agent_manager/head` stores a namespace name (plain text). When a
request arrives at `/head/`, the gateway reads this file and proxies the request
to the matching registered namespace. The file is written by the `POST
/api/gateway/head` endpoint, which the dashboard **"Set as HEAD"** button calls.

If the file is missing or points to an unregistered namespace, the gateway falls
back to the first namespace returned by `reg.List()`.

## Adding a new feature

**New API endpoint in the sidecar**: add a handler in `sidecar/main.go` and
register it in `registerRoutes()`. The gateway automatically proxies `/api/<path>`
to the sidecar for all workspace prefixes.

**New gateway-internal endpoint**: add a `HandleFunc` in `gateway/server.go`
before the catch-all `"/"` handler. Gateway-internal paths should be under
`/api/gateway/` to avoid collisions with sidecar paths.

**New CLI subcommand**: add a `cmd_<name>` function in `agent_manager.py` and
register it in `main()` with `sub.add_parser(...)`.

**New namespace feature** (parallel branch testing): see [workspace.md](workspace.md).
