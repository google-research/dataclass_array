# Gateway WorkspaceEntry Redesign — Abstraction Design

## Problem statement

The Go gateway still uses the old `WorkspaceEntry` shape (`path`, `bundle_path`,) while the Python CLI was updated to the new three-path model
(`ls_path`, `sidecar_path`, `frontend_path`). The two are out of sync.

Beyond the schema mismatch, the current design scatters merge-and-fallback
logic across `Registry`, `HeadPointer`, and `Supervisor`. This redesign
extracts that into one self-contained abstraction.

---

## New data model

### `WorkspaceInfo` — the shared, nullable three-path record

```go
// WorkspaceInfo is the JSON-serialisable shape shared with the Python CLI.
// Every field is optional: only the paths that have been built are present.
type WorkspaceInfo struct {
    LSPath       *string `json:"ls_path,omitempty"`
    SidecarPath  *string `json:"sidecar_path,omitempty"`
    FrontendPath *string `json:"frontend_path,omitempty"`
}
```

- Mirrors the Python `WorkspaceEntry` field-for-field (minus `name`, which is
  the filename key).
- `nil` pointer = "not built for this workspace yet".
- Serialises with `omitempty` so absent fields are truly absent in JSON.

### `WorkspaceEntry` — identity + info

```go
type WorkspaceEntry struct {
    Name string        // derived from filename, never serialised
    Info WorkspaceInfo // raw (non-merged) data from this workspace's JSON
}
```

The `Name` field is NOT written to the JSON file (it is the filename stem),
matching Python's `to_dict()` which already omits it.

---

## Merge semantics

```
Merged(workspace, head) = workspace.Info, with nil fields filled from head.Info
```

```go
// Merge fills nil fields in base with values from fallback.
// Returns a new WorkspaceInfo; neither input is mutated.
func (base WorkspaceInfo) Merge(fallback WorkspaceInfo) WorkspaceInfo {
    out := base
    if out.LSPath == nil {
        out.LSPath = fallback.LSPath
    }
    if out.SidecarPath == nil {
        out.SidecarPath = fallback.SidecarPath
    }
    if out.FrontendPath == nil {
        out.FrontendPath = fallback.FrontendPath
    }
    return out
}
```

This is pure — no I/O, no side effects, easy to test.

---

## `WorkspaceStore` — self-contained workspace management abstraction

The `WorkspaceStore` owns everything that touches the workspace JSON files.
It replaces the current `Registry` + `HeadPointer.Resolve()` responsibilities.

```go
type WorkspaceStore struct {
    dir      string        // ~/.agent_manager/workspaces/
    headPath string        // ~/.agent_manager/head.json  (not "head", see below)

    mu        sync.RWMutex
    cache     map[string]*WorkspaceEntry
    cacheTime time.Time
    cacheTTL  time.Duration
}
```

### Key methods

```go
// List returns all registered workspace entries (raw, non-merged).
func (s *WorkspaceStore) List() []*WorkspaceEntry

// Get returns a workspace entry by name (raw, non-merged), or nil.
func (s *WorkspaceStore) Get(name string) *WorkspaceEntry

// Resolve returns the effective WorkspaceInfo for a workspace,
// with nil fields filled from HEAD.
// Returns nil only if the workspace does not exist.
func (s *WorkspaceStore) Resolve(name string) *WorkspaceEntry

// Head returns the current HEAD WorkspaceEntry (raw), or nil.
func (s *WorkspaceStore) Head() *WorkspaceEntry

// ResolvedHead returns the HEAD entry with nil fields filled from... itself.
// (HEAD is always complete, so this is mostly a passthrough.)
func (s *WorkspaceStore) ResolvedHead() *WorkspaceEntry

// SetHead atomically writes name as the head and updates head.json.
func (s *WorkspaceStore) SetHead(name string) error
```

### `head.json` — always complete

HEAD is stored as a JSON file (`~/.agent_manager/head.json`) rather than a
plain text pointer file. It contains:

1. `name` — which workspace is HEAD.
2. A complete `WorkspaceInfo` — the **merged** info at the time it was last
   updated (all three paths present).

```json
{
  "name": "refactor-head-to-gateway",
  "ls_path":       "/path/to/language_server",
  "sidecar_path":  "/path/to/sidecar",
  "frontend_path": "/path/to/dist"
}
```

This means:
- HEAD is **always** the fallback source of truth.
- If a workspace only built its sidecar, the gateway fills in LS and frontend
  from HEAD automatically.

### Updating HEAD

When a workspace becomes HEAD (`SetHead`):

```
head.Info = head.Info.Merge(workspace.Info)
```

i.e. the workspace's paths *override* the old HEAD values, but only for the
paths that are present. The remaining paths stay as they were in HEAD.

This keeps HEAD always complete (assuming it was complete when first populated).

---

## How Supervisor changes

The current `Supervisor` hard-codes binary path discovery logic (blaze-bin
paths). That logic moves to `WorkspaceStore.Resolve()` — the supervisor just
receives a resolved `WorkspaceInfo` and reads the paths directly.

```go
func (s *Supervisor) GetOrStartSidecar(workspace *WorkspaceEntry) (int, error) {
    info := store.Resolve(workspace.Name)
    if info.SidecarPath == nil {
        return 0, fmt.Errorf("sidecar not built for %s", workspace.Name)
    }
    return s.getOrStart(workspace.Name, "sidecar", *info.SidecarPath, ...)
}
```

The `Supervisor.UpdateDefaultsFromWorkspace()` method is deleted — it is
replaced by the `Resolve()` call above, which always fetches the correct
fallback from HEAD.

---

## What changes in the gateway `server.go`

The `server.go` routing code now calls `store.Resolve(name)` instead of
`reg.Get(name)` when it needs the effective binary paths. The routing itself
(workspace prefix → which workspace) still uses `store.Get()` for existence
checks.

---

## File layout (proposed)

```
gateway/
  workspace.go      # WorkspaceInfo, WorkspaceEntry, Merge()
  store.go          # WorkspaceStore (replaces registry.go + head.go)
  supervisor.go     # updated to use WorkspaceStore
  server.go         # updated to use WorkspaceStore
  proxy.go          # unchanged
  main.go           # updated wiring
```

---

## Issues I spotted that you may have overlooked

### 1. `head.json` vs plain `head` text file
The current gateway stores HEAD as a plain text file (workspace name only).
The new design proposes `head.json` to store the complete merged info alongside
the name. **Migration needed**: on startup, if `head.json` is missing but
`head` (plain text) exists, the gateway should migrate it.

### 2. Head bootstrap problem
If `head.json` does not exist yet (first run), HEAD paths are all nil. The
system cannot start. You need either:
  - A `--head=<workspace>` flag on `./gateway` that populates HEAD from a
    registered workspace.
  - Or the Python CLI's `start` command explicitly calls `SetHead` before
    launching the gateway.

Currently `--set-workspace-head` does this, but only if the workspace is
already registered. That flow will need to be preserved.

### 3. Frontend path vs bundle path
The current gateway uses `BundlePath` (passed to the LS command) and `Path`
(the worktree root). In the new model, `FrontendPath` replaces both — it is
the compiled `dist/` directory, which is also the bundle the LS serves.
Double-check that the LS `--jetbox_bundle_path` flag expects the `dist/`
directory (not the worktree root).

### 4. The `frontend_path` typo in the user story
The user's description uses `frontent_path` (missing 'd') — Python registry
uses `frontend_path`. This design uses `frontend_path` to match Python.

### 5. Reaper / idle timeout still needs `WorkspaceEntry.Name`
The reaper accesses instances by name. This is unaffected by the redesign
(`WorkspaceEntry.Name` remains the key).

### 6. `handleListWorkspaces` response shape
The current endpoint returns the raw `WorkspaceEntry` array. After the
redesign, do you want the API to return raw or resolved entries? Recommendation:
return raw per-workspace data plus HEAD info separately so the frontend can
show what each workspace has built.

---

## Verification plan

### Unit tests
- `workspace_test.go`: pure `Merge()` tests — nil propagation, all combos.
- `store_test.go`: replaces `registry_test.go` + `head_test.go`.
  - Write workspace JSON with partial fields, read back, assert merge.
  - SetHead → assert head.json written correctly.

### Integration
- Build sidecar only → workspace entry has only `sidecar_path`.
- Gateway resolves it: sidecar comes from workspace, LS + frontend from HEAD.
- Build frontend → workspace entry gains `frontend_path`.
- Gateway now serves workspace frontend, falls back sidecar/LS from HEAD.
