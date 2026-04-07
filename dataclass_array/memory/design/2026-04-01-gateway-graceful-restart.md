# Gateway Graceful Restart (Pragmatic Approach)

**Status:** Draft
**Date:** 2026-04-01

## Problem

When the gateway is restarted (e.g. after `./agent_manager.py restart --build-gateway`), it
kills the Language Server and Sidecar processes it supervises. Any agent session running _inside_
that gateway — including the one that triggered the restart — is interrupted. The browser must wait
several seconds for all child processes to start again before the UI is usable.

### Why the gateway needs restarting at all

The gateway binary is only rebuilt when the Go source under `gateway/` changes. Frontend, Sidecar,
and LS changes are hot-deployed without any restart. Gateway restarts are therefore infrequent, but
when they happen, they are currently disruptive enough that the developer must manually re-open the
browser tab and wait for the stack to come back up.

## Goals

- The Language Server and Sidecar processes survive a gateway restart.
- After the new gateway is up, the browser reconnects within ~100 ms (a single TCP reconnect).
- Existing streaming RPC connections (gRPC-web) are reset once at the TCP level but immediately
  re-established by the browser; the LS session itself is intact.
- No architectural complexity beyond what a single-developer tool warrants.

## Non-Goals

- True zero-downtime (no TCP connection drop). This requires file-descriptor passing between
  processes (`SCM_RIGHTS`) and is not worth the complexity for a dev tool.
- Preserving the supervisor's in-memory crash history across restarts.
- Full supervision (auto-restart, crash detection) of adopted child processes. Adopted processes
  are unmanaged after re-adoption; they must be explicitly restarted via the UI or CLI if they
  crash.

## Design

### Invariant: children outlive the gateway

The key change is that **the gateway does not kill its children on exit**. Today, `sup.Stop()`
sends `SIGINT` to all supervised processes before `os.Exit(0)`. After this change, the gateway
exits without touching its children. The OS re-parents them to PID 1 (init), where they continue
running undisturbed.

### New abstraction: `RuntimeStore`

All the state needed to re-adopt an orphaned child is written to disk _at spawn time_, before the
child is started. The data lives in `~/.agent_manager/runtime/<namespace>/` and is fully managed
by a new, self-contained module: `gateway/runtime_store.go`.

```
~/.agent_manager/
  namespaces/          ← existing: namespace build paths (NamespaceInfo JSON)
  runtime/             ← new: per-namespace live process state
    head/
      sidecar.json     ← {pid, port, csrf_token}  (csrf_token absent for sidecar)
      language_server.json
    feat-x/
      sidecar.json
      language_server.json
```

`RuntimeStore` is the only component that reads or writes these files. Its public surface is
minimal:

```go
// ServiceRecord is the persisted state of one running child process.
// It is written before the child starts and deleted when the child is stopped
// intentionally (i.e. not on gateway exit).
type ServiceRecord struct {
    PID       int    `json:"pid"`
    Port      int    `json:"port"`
    CSRFToken string `json:"csrf_token,omitempty"` // only Language Server
}

type RuntimeStore struct { dir string }

func NewRuntimeStore(dir string) (*RuntimeStore, error)

// Write atomically writes a record for (namespace, service).
func (r *RuntimeStore) Write(namespace, service string, rec ServiceRecord) error

// Read returns the persisted record for (namespace, service), or nil if absent.
func (r *RuntimeStore) Read(namespace, service string) (*ServiceRecord, error)

// Delete removes the record for (namespace, service).
func (r *RuntimeStore) Delete(namespace, service string) error

// ReadAll returns all records grouped by namespace name.
func (r *RuntimeStore) ReadAll() (map[string]map[string]ServiceRecord, error)
```

`RuntimeStore` has no dependency on the `Supervisor` or `NamespaceStore`. It is a pure persistence
layer for `ServiceRecord` values.

### Supervisor changes: write before spawn, read on startup

#### On spawn

`Supervisor.getOrStartProcWithBinPath` writes a `ServiceRecord` to `RuntimeStore` _before_
calling `cmd.Start()`. If `cmd.Start()` fails, the record is deleted.

The CSRF token is generated once per LS spawn and stored in the `ServiceRecord`. The token is read
back from the record during re-adoption so the new gateway can proxy to the unchanged LS correctly.

`buildLSCmd` is updated to accept the token as a parameter instead of generating it internally.
`generateCSRFToken` remains in `main.go` and is called by the Supervisor before spawning.

#### On shutdown

`sup.Stop()` is **removed** from the signal handler. The gateway exits cleanly without sending any
signal to children. `RuntimeStore` records are intentionally _not_ deleted on exit — they are the
handoff to the next gateway instance.

Records _are_ deleted when the Supervisor intentionally stops a namespace (e.g.
`RestartNamespace`), since in that case the children are killed on purpose.

#### On startup: `Supervisor.AdoptOrphans`

A new method is called once, during gateway startup, after the `Supervisor` is initialised:

```go
// AdoptOrphans reads all ServiceRecords from the RuntimeStore and attempts to
// re-adopt any processes that are still alive (verified via kill(pid, 0)).
// Adopted instances are registered in the supervisor's instances map with
// IsAdopted=true. Stale records (dead PID) are deleted from the RuntimeStore.
func (s *Supervisor) AdoptOrphans(rs *RuntimeStore) error
```

For each live record, `AdoptOrphans`:

1. Creates a `NamespaceInstance` and `serviceInstance` in the supervisor's `instances` map.
2. Sets `svc.Port` and `svc.BinPath` from the record — the port is what matters for proxying.
3. Sets `svc.Proc = nil` (no `exec.Cmd` — we did not spawn the process).
4. Sets `svc.IsAdopted = true` (new field on `serviceInstance`).
5. Does **not** start a `supervise()` goroutine — adopted processes are unmanaged.

The proxy (`namespaceProxy`) is constructed immediately from the adopted ports, so the first
browser request after startup is served without any cold-start delay.

### `serviceInstance` gets `IsAdopted bool`

```go
type serviceInstance struct {
    Proc      *managedProc
    Port      int
    BinPath   string
    IsAdopted bool // true when process was inherited from a previous gateway
}
```

`IsAdopted` is surfaced in `ServiceStatus` (the JSON struct returned by `/api/gateway/status`):

```go
type ServiceStatus struct {
    Name      string       `json:"name"`
    Port      int          `json:"port"`
    PID       int          `json:"pid"`
    Up        bool         `json:"up"`
    BinPath   string       `json:"bin_path"`
    IsAdopted bool         `json:"is_adopted"` // new
    Crashes   []CrashEvent `json:"crashes"`
}
```

The UI can use `is_adopted` to display a visual indicator (e.g. "⟳ adopted" badge next to the
sidecar/LS status) with a "Restart" button that calls the existing
`POST /api/gateway/restart_from_head` endpoint to replace the adopted processes with fresh,
fully-supervised ones.

### `processUpAndPID` for adopted processes

`processUpAndPID` currently inspects `svc.Proc.proc.Process`. For adopted processes, `svc.Proc`
is nil. A new helper `adoptedProcessUp(pid int) bool` uses `syscall.Kill(pid, 0)` to check
liveness. `processUpAndPID` is updated to fall back to this when `IsAdopted` is true.

### Startup sequence (updated)

```
main()
  ├─ NewNamespaceStore(...)        // unchanged
  ├─ NewRuntimeStore(...)          // new
  ├─ sup.Start(ctx)                // unchanged
  ├─ sup.AdoptOrphans(runtimeStore) // new: re-adopt live children
  ├─ sup.StartReaper(ctx, ...)     // unchanged
  ├─ registerRoutes(mux, store, sup) // unchanged
  └─ srv.ListenAndServe()          // unchanged
```

### Shutdown sequence (updated)

```
SIGTERM / Ctrl-C
  └─ cancel()  // cancel context → supervise() goroutines exit
               // children are NOT killed
               // RuntimeStore records are NOT deleted
               // os.Exit(0) (or srv.Shutdown)
```

### Port binding: `SO_REUSEPORT`

To eliminate the ~100 ms TCP gap during the handover window, the gateway binds port 3001 with
`SO_REUSEPORT`. This allows the new gateway process to start accepting connections before the old
one has fully exited. In practice the gap is zero at the kernel level; the browser experiences no
connection error.

Implementation: replace `srv.ListenAndServe()` with a manually created `net.Listener` using
`net.ListenConfig` with `Control` set to apply `SO_REUSEPORT` via `syscall.SetsockoptInt`.

This is a small, self-contained change in `main.go` (~15 lines).

### `agent_manager.py restart` sequence (updated)

```
1. build (optional, --build-gateway etc.)
2. send SIGTERM to old gateway   ← gateway exits; children survive
3. wait for port 3001 to be free (with SO_REUSEPORT this may be instant)
4. start new gateway
5. new gateway calls AdoptOrphans → proxy is ready immediately
6. browser reconnects → requests served without cold-start
```

Steps 2–4 already exist in `cmd_stop` / `cmd_start`. No Python changes are required.

## File Map

| File | Change |
|---|---|
| `gateway/runtime_store.go` | **New.** `RuntimeStore`, `ServiceRecord`. |
| `gateway/supervisor.go` | Add `IsAdopted` to `serviceInstance`; add `AdoptOrphans`; write/delete `ServiceRecord` on spawn/stop; remove `SIGINT` fan-out from `Stop()`. |
| `gateway/main.go` | Add `NewRuntimeStore`; call `AdoptOrphans`; switch to `SO_REUSEPORT` listener; remove `sup.Stop()` from signal handler. |
| `gateway/server.go` | Expose `is_adopted` in `ServiceStatus`. |
| `gateway/namespace.go` | No change. |
| `gateway/store.go` | No change. |
| `gateway/proxy.go` | No change. |

## Known Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| Adopted processes have no auto-restart | If the LS crashes after adoption, it stays down silently | `is_adopted` UI badge + "Restart" button; user calls `restart_from_head` |
| Adopted processes are not killed on new gateway exit | If two consecutive restarts happen before the user clicks "Restart", stale children accumulate | `AdoptOrphans` cleans up dead PIDs; `restart_from_head` kills and respawns |
| Crash history is lost | Observability gap | Accepted; crash history is a debug aid, not critical |
| `SO_REUSEPORT` is Linux-only | N/A for this use case (Cloud Workstations) | None needed |
| Streaming gRPC-web connections reset once | Single reconnect spinner | Accepted; LS session is intact |

## Alternatives Considered

### Full zero-downtime (file-descriptor passing)

Pass the listening socket's file descriptor from old to new gateway via a Unix socket using
`SCM_RIGHTS`. The new gateway inherits the live listener with zero TCP gap. Rejected: ~1 week of
implementation, adds significant complexity to `main.go`, not warranted for a dev tool.

### Killing children but restarting them faster

Keep the current kill-on-exit behaviour but make the LS start faster. Not viable: the LS cold
start is ~3–5 s and is controlled by the Antigravity team's binary, not ours.

### Nginx / Caddy as the outer proxy

Replacing the Go gateway with a standard reverse proxy would give us graceful reload for free
(`nginx -s reload`). Rejected: the gateway's namespace-routing logic, referer fallback, and
supervisor are not expressible in Nginx config without custom Lua modules. See earlier research
(conversation `e81d3927`).
