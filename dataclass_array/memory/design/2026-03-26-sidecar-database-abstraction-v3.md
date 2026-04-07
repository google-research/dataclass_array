# Sidecar Database Abstraction

> **Status**: Design draft — 2026-03-26 (rev 3)
>
> **Scope**: Go sidecar backend + frontend data flow.
>
> **Starting point**: fresh — no v1 migration.
>
> **First use-case**: TODO Manager plugin.

--------------------------------------------------------------------------------

## 1. Problem Statement

The v2 sidecar returns stub empty arrays. We need a real persistence layer.
Constraints:

-   **All workspaces share one PostgreSQL database.** Each workspace has its own
    sidecar process, but all connect to the same Postgres. Concurrency is a
    first-class concern.
-   **`Conversation` is a sidecar-owned core concept.** Any plugin can annotate
    a conversation with plugin-specific metadata. No plugin owns or locks it.
-   **Plugin data is self-contained.** Each plugin brings its own schema SQL and
    HTTP routes. The sidecar core doesn't know the plugin internals.

--------------------------------------------------------------------------------

## 2. Goals and Non-Goals

**Goals**

-   `Conversation` is a core, generic, plugin-agnostic entity.
-   Multiple plugins can independently annotate the same conversation.
-   The response for a conversation merges all plugin annotations into a
    `plugins` map: `{"todo-manager": {...}, "other-plugin": {...}}`.
-   Clean plugin registration: each plugin contributes schema SQL + HTTP routes
    +   a `GetConversationMeta` hook.
-   No wrapping of `pgx` — pass `*pgxpool.Pool` directly.
-   Idempotent embedded SQL schema, advisory-lock protected.
-   Gateway owns Postgres lifecycle (it already owns sidecar and LS).

**Non-Goals**

-   No `Database` interface wrapping `pgxpool.Pool`.
-   No ORM, no generic CRUD base.
-   No real-time push — polling is fine.
-   No v1 data import.

--------------------------------------------------------------------------------

## 3. Architecture Overview

```
┌── Browser (React) ────────────────────────────────────────────────────────┐
│  fetch('/api/conversations')          → list + merged plugin metadata     │
│  fetch('/api/conversations/:id')      → single with full plugin map       │
│  fetch('/api/todo-manager/todos')     → plugin-specific endpoint          │
└────────────────────────────────────────────────────────────────────────────┘
              │ HTTP absolute path — Gateway routes by request metadata
┌── Gateway ────────────────────────────────────────────────────────────────┐
│  Owns: sidecar lifecycle, LS lifecycle, Postgres lifecycle                │
│  Routes /head/api/* → sidecar :3000                                      │
└───────────────────┬────────────────────────────────────────────────────────┘
                    │ proxy (prefix stripped)
┌── Sidecar ─────────────────────────────────────────────────────────────────┐
│  main.go                                                                   │
│   ├── db.Open(dsn) → *pgxpool.Pool                                         │
│   └── plugin.Registry → registers plugins, applies schema, wires routes    │
│        │                                                                    │
│        ├── /api/health                → healthHandler                       │
│        ├── /api/conversations         → ConversationHandler(registry)       │ ← core
│        ├── /api/conversations/:id     → ConversationHandler(registry)       │ ← core
│        └── /api/todo-manager/todos/*  → TodoManagerHandler(pool)            │ ← plugin
│                                                                             │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │ pgx pool
┌── PostgreSQL (shared) ──────▼────────────────────────────────────────────────┐
│  conversations                     (sidecar core)                           │
│  todo_manager_todos                (todo-manager plugin)                    │
│  todo_manager_conversation_meta    (todo-manager plugin)                    │
└──────────────────────────────────────────────────────────────────────────────┘
       ▲                 ▲                 ▲
 workspace A        workspace B       workspace C
 (sidecar proc)     (sidecar proc)    (sidecar proc)
```

--------------------------------------------------------------------------------

## 4. Database Package — No `pgx` Wrapper

`*pgxpool.Pool` is the concrete type used everywhere. No wrapper struct, no
interface.

```go
// sidecar/db/db.go
package db

// Open creates a connection pool and applies the embedded core schema.
// Also calls plugin.ApplySchema for each registered plugin.
// The advisory lock inside applySchema makes this safe for concurrent callers.
func Open(ctx context.Context, dsn string, plugins []plugin.Plugin) (*pgxpool.Pool, error) {
    pool, err := pgxpool.New(ctx, dsn)
    if err != nil {
        return nil, fmt.Errorf("open pool: %w", err)
    }
    if err := ApplySchema(ctx, pool, plugins); err != nil {
        pool.Close()
        return nil, fmt.Errorf("apply schema: %w", err)
    }
    return pool, nil
}
```

Cross-cutting concerns (logging, tracing) belong in a `pgx.QueryTracer` set on
the pool config — pgx's own extension point.

--------------------------------------------------------------------------------

## 5. Schema Strategy

Single idempotent `CREATE TABLE IF NOT EXISTS` script — no migration library.
We're starting fresh; versioned migrations pay off when you need to ALTER a live
schema without a drop, which we don't have yet.

Advisory lock during schema application ensures multiple sidecars racing at
startup don't interleave DDL:

```go
// sidecar/db/schema.go
func ApplySchema(ctx context.Context, pool *pgxpool.Pool, plugins []plugin.Plugin) error {
    conn, err := pool.Acquire(ctx)
    if err != nil { return err }
    defer conn.Release()

    const lockKey = int64(0x616d736368656d61) // "amschema"
    if _, err := conn.Exec(ctx, "SELECT pg_advisory_lock($1)", lockKey); err != nil {
        return fmt.Errorf("advisory lock: %w", err)
    }
    defer conn.Exec(ctx, "SELECT pg_advisory_unlock($1)", lockKey) //nolint:errcheck

    // Core schema (conversations table).
    if _, err := conn.Exec(ctx, coreSchemaSQL); err != nil {
        return fmt.Errorf("core schema: %w", err)
    }
    // Each plugin's schema.
    for _, p := range plugins {
        if sql := p.SchemaSQL(); sql != "" {
            if _, err := conn.Exec(ctx, sql); err != nil {
                return fmt.Errorf("plugin %s schema: %w", p.ID(), err)
            }
        }
    }
    return nil
}
```

--------------------------------------------------------------------------------

## 6. Plugin Interface

Each plugin is a self-contained unit that contributes:

1.  **Schema SQL** — its own tables (embedded, idempotent).
2.  **HTTP routes** — registered on the sidecar's mux.
3.  **Conversation metadata hook** — queried when serving `GET
    /api/conversations/:id`.

```go
// sidecar/plugin/plugin.go
package plugin

import (
    "context"
    "net/http"

    "github.com/jackc/pgx/v5/pgxpool"
)

// Plugin is implemented by each sidecar plugin.
type Plugin interface {
    // ID returns the plugin's unique identifier (e.g. "todo-manager").
    ID() string

    // SchemaSQL returns idempotent DDL SQL for this plugin's tables.
    // Called once at startup under the schema advisory lock.
    SchemaSQL() string

    // RegisterRoutes wires this plugin's HTTP handlers onto mux.
    // Called once at startup after schema is applied.
    RegisterRoutes(mux *http.ServeMux)

    // GetConversationMeta returns the plugin's annotation for a given
    // conversation ID, or nil if this plugin has no data for it.
    // Called on every GET /api/conversations/:id.
    GetConversationMeta(ctx context.Context, conversationID string) (any, error)
}

// Registry holds all registered plugins and provides batch operations.
type Registry struct {
    plugins []Plugin
}

func NewRegistry(plugins ...Plugin) *Registry {
    return &Registry{plugins: plugins}
}

func (r *Registry) All() []Plugin { return r.plugins }

// GetAllConversationMeta calls each plugin's GetConversationMeta and returns
// the merged map. Plugins with nil results are omitted.
func (r *Registry) GetAllConversationMeta(ctx context.Context, conversationID string) (map[string]any, error) {
    result := make(map[string]any)
    for _, p := range r.plugins {
        meta, err := p.GetConversationMeta(ctx, conversationID)
        if err != nil {
            return nil, fmt.Errorf("plugin %s: %w", p.ID(), err)
        }
        if meta != nil {
            result[p.ID()] = meta
        }
    }
    return result, nil
}
```

--------------------------------------------------------------------------------

## 7. Core: `Conversation`

A `Conversation` is identified directly by the **Antigravity cascade ID** — the same
ID that appears in the URL as `/c/<id>`. There is no separate sidecar-assigned
UUID: the Antigravity ID is the primary key.

### 7.1 Go type

```go
// sidecar/store/conversation.go
package store

import "time"

// Conversation is the sidecar's core record for any Antigravity conversation
// launched from any plugin. Plugin-specific data lives in extension tables
// that reference this ID.
//
// ID and the Antigravity cascade ID are unified — the Antigravity ID is the PK.
type Conversation struct {
    ID          string    // Antigravity cascade ID (/c/<id>)
    WorkspaceID string    // which workspace launched this
    CreatedAt   time.Time
}

// ConversationWithPlugins is the full response object served to the frontend.
type ConversationWithPlugins struct {
    Conversation
    Plugins map[string]any `json:"plugins"` // keyed by plugin ID
}
```

`Mode` (`navigate` / `direct`) is **not stored** — it is transient UI state used
only at conversation-creation time to decide how to open it in the frontend.
Once the conversation exists, it doesn't matter.

### 7.2 Core schema

```sql
-- sidecar/db/schema.sql  (core only — embedded in db package)

CREATE TABLE IF NOT EXISTS conversations (
    id           TEXT        PRIMARY KEY,   -- Antigravity cascade ID
    workspace_id TEXT        NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_workspace_idx ON conversations(workspace_id);
```

### 7.3 Repository

```go
// sidecar/store/conversation_repo.go
package store

type ConversationRepo struct{ pool *pgxpool.Pool }

func NewConversationRepo(pool *pgxpool.Pool) *ConversationRepo {
    return &ConversationRepo{pool: pool}
}

func (r *ConversationRepo) Create(ctx context.Context, id, workspaceID string) (Conversation, error) {
    var c Conversation
    err := r.pool.QueryRow(ctx,
        `INSERT INTO conversations (id, workspace_id)
         VALUES ($1, $2)
         ON CONFLICT (id) DO NOTHING
         RETURNING id, workspace_id, created_at`,
        id, workspaceID,
    ).Scan(&c.ID, &c.WorkspaceID, &c.CreatedAt)
    // ON CONFLICT handles the race where two sidecars register the same cascade ID.
    if err != nil {
        return Conversation{}, fmt.Errorf("insert conversation %s: %w", id, err)
    }
    return c, nil
}

func (r *ConversationRepo) ListAll(ctx context.Context) ([]Conversation, error) {
    rows, err := r.pool.Query(ctx,
        `SELECT id, workspace_id, created_at FROM conversations ORDER BY created_at DESC`)
    if err != nil {
        return nil, fmt.Errorf("list conversations: %w", err)
    }
    defer rows.Close()
    return pgx.CollectRows(rows, pgx.RowToStructByName[Conversation])
}

func (r *ConversationRepo) Get(ctx context.Context, id string) (Conversation, error) {
    var c Conversation
    err := r.pool.QueryRow(ctx,
        `SELECT id, workspace_id, created_at FROM conversations WHERE id=$1`, id,
    ).Scan(&c.ID, &c.WorkspaceID, &c.CreatedAt)
    if errors.Is(err, pgx.ErrNoRows) {
        return Conversation{}, ErrNotFound
    }
    return c, err
}
```

### 7.4 Core HTTP handler

```
POST /api/conversations            → create conversation record
GET  /api/conversations            → list all conversations
GET  /api/conversations/:id        → single conversation + merged plugin annotations
```

The list endpoint returns `ConversationWithPlugins` for each conversation. For
efficiency, plugins are queried **in batch per plugin**, not per conversation
(avoids N+1):

```go
// GET /api/conversations
func (h *conversationHandler) handleList(w http.ResponseWriter, r *http.Request) {
    convos, _ := h.repo.ListAll(ctx)

    // Batch: for each plugin, fetch all meta in one query.
    pluginMetas := make(map[string]map[string]any) // pluginID → conversationID → meta
    for _, p := range h.registry.All() {
        metas, _ := p.BatchGetConversationMeta(ctx, convIDs(convos))
        pluginMetas[p.ID()] = metas
    }

    // Assemble response.
    var result []ConversationWithPlugins
    for _, c := range convos {
        plugins := map[string]any{}
        for pluginID, metas := range pluginMetas {
            if m, ok := metas[c.ID]; ok {
                plugins[pluginID] = m
            }
        }
        result = append(result, ConversationWithPlugins{Conversation: c, Plugins: plugins})
    }
    writeJSON(w, http.StatusOK, result)
}
```

This adds a second method to the `Plugin` interface:

```go
// BatchGetConversationMeta returns metadata for multiple conversations at once.
// Returns a map of conversationID → metadata. Missing IDs are omitted.
BatchGetConversationMeta(ctx context.Context, ids []string) (map[string]any, error)
```

--------------------------------------------------------------------------------

## 8. Plugin-Specific Data: Extension Tables

Each plugin creates its own tables. Schema SQL lives **in the plugin folder**.
No shared metadata blob — typed extension tables are safer and queryable.

### 8.1 TODO Manager schema

```sql
-- sidecar/todo_manager/schema.sql  (embedded in todo_manager package)

CREATE TYPE IF NOT EXISTS todo_status AS ENUM (
    'not_started',
    'in_progress',
    'needs_review',
    'done'
);

CREATE TABLE IF NOT EXISTS todo_manager_todos (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    title      TEXT        NOT NULL,
    parent_id  UUID        REFERENCES todo_manager_todos(id) ON DELETE CASCADE,
    status     todo_status NOT NULL DEFAULT 'not_started',
    content    TEXT        NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- A todo can have many conversations (multiple attempts).
-- A conversation can be annotated by this plugin at most once.
CREATE TABLE IF NOT EXISTS todo_manager_conversation_meta (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    todo_id         UUID NOT NULL REFERENCES todo_manager_todos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS todo_meta_todo_idx ON todo_manager_conversation_meta(todo_id);
```

Key decisions: - `status` is a PostgreSQL `ENUM` — invalid values are rejected
at the DB level. - One conversation → at most one todo (via PK on
`conversation_id`). - One todo → many conversations (no unique constraint on
`todo_id`).

### 8.2 TODO Manager Go types

```go
// sidecar/todo_manager/types.go
package todomanager

import "time"

type Todo struct {
    ID        string    `db:"id"`
    Title     string    `db:"title"`
    ParentID  *string   `db:"parent_id"`
    Status    string    `db:"status"`
    Content   string    `db:"content"`
    CreatedAt time.Time `db:"created_at"`
    UpdatedAt time.Time `db:"updated_at"`
    Children  []Todo    `db:"-"` // assembled in-process from flat list
}

// ConversationMeta is the plugin's annotation for a single conversation.
// This is what appears under "todo-manager" in the plugins map.
type ConversationMeta struct {
    TodoID string `json:"todoId"`
    Todo   *Todo  `json:"todo,omitempty"` // populated on request
}
```

### 8.3 TODO Manager Plugin implementation

```go
// sidecar/todo_manager/plugin.go
package todomanager

import (
    _ "embed"
    "sidecar/plugin"
)

//go:embed schema.sql
var schemaSQL string

type TodoManagerPlugin struct {
    repo *Repo
}

func New(pool *pgxpool.Pool) *TodoManagerPlugin {
    return &TodoManagerPlugin{repo: NewRepo(pool)}
}

func (p *TodoManagerPlugin) ID() string        { return "todo-manager" }
func (p *TodoManagerPlugin) SchemaSQL() string { return schemaSQL }

func (p *TodoManagerPlugin) RegisterRoutes(mux *http.ServeMux) {
    h := newHandler(p.repo)
    mux.HandleFunc("/api/todo-manager/todos", h.handleTodos)
    mux.HandleFunc("/api/todo-manager/todos/", h.handleTodo)
}

func (p *TodoManagerPlugin) GetConversationMeta(ctx context.Context, id string) (any, error) {
    meta, err := p.repo.GetConversationMeta(ctx, id)
    if errors.Is(err, store.ErrNotFound) {
        return nil, nil // this plugin has no data for this conversation
    }
    return meta, err
}

func (p *TodoManagerPlugin) BatchGetConversationMeta(ctx context.Context, ids []string) (map[string]any, error) {
    metas, err := p.repo.BatchGetConversationMeta(ctx, ids)
    if err != nil {
        return nil, err
    }
    result := make(map[string]any, len(metas))
    for k, v := range metas {
        result[k] = v
    }
    return result, nil
}
```

--------------------------------------------------------------------------------

## 9. Concurrency: Multiple Sidecars, One Database

| Situation                         | Mechanism                                |
| --------------------------------- | ---------------------------------------- |
| Schema DDL at startup (N sidecars | `pg_advisory_lock("amschema")` — only    |
: racing)                           : one runs DDL, others wait and then no-op :
:                                   : on `IF NOT EXISTS`                       :
| Inserting a new conversation      | `ON CONFLICT (id) DO NOTHING` — two      |
:                                   : sidecars registering the same Antigravity ID  :
:                                   : is safe                                  :
| Creating a todo                   | Plain `INSERT` — no two sidecars create  |
:                                   : the same todo                            :
| Updating a todo (optimistic)      | `UPDATE ... WHERE id=$1 AND              |
:                                   : updated_at=$2` — returns 0 rows if       :
:                                   : stale; caller gets 409                   :
| `status` validity                 | `ENUM` constraint enforced at DB level — |
:                                   : no invalid status can be written         :
| Read-then-mutate (e.g. reparent   | `SELECT ... FOR UPDATE` inside explicit  |
: subtree)                          : `BEGIN`/`COMMIT`                         :

No `sync.Mutex` in repositories — that was the v1 workaround that only helped
within one process.

--------------------------------------------------------------------------------

## 10. Postgres Lifecycle

**The Gateway owns Postgres.** It already supervises the sidecar and Language
Server; Postgres is another supervised child in the same pattern.

```
Gateway startup:
  1. Start Postgres  (or attach to an already-running instance)
  2. Wait for Postgres to accept connections
  3. Start sidecar  (sidecar calls db.Open → pool + schema)
  4. Start LS
```

The DSN (Unix socket path or TCP address) is passed to the sidecar via a
`--db_dsn` flag, just as `--port` is passed today. The sidecar does not manage
Postgres — it only opens a pool to an already-running instance.

--------------------------------------------------------------------------------

## 11. Frontend Data Flow

### URL structure

```
/api/conversations           ← sidecar core (generic, all plugins)
/api/conversations/:id       ← single conversation + plugin map
/api/todo-manager/todos/*    ← todo-manager plugin (todos only)
```

### Absolute paths

Frontend code uses absolute paths (`fetch('/api/conversations')`). The gateway
routes requests to the correct sidecar using request metadata (e.g. Referer
header or session cookie that identifies the workspace). No workspace prefix
embedded in the fetch URL.

> **Open design question**: the exact mechanism by which the gateway identifies
> the workspace from an absolute `/api/` request needs to be specified in the
> gateway design doc. This is intentionally deferred.

### Response shape

```ts
// GET /api/conversations
type ConversationList = ConversationWithPlugins[];

// GET /api/conversations/:id
interface ConversationWithPlugins {
    id: string;          // Antigravity cascade ID
    workspaceId: string;
    createdAt: string;   // ISO 8601
    plugins: {
        "todo-manager"?: {
            todoId: string;
            todo?: { id: string; title: string; status: string; ... };
        };
        // other plugins...
    };
}
```

### How the React store uses this

```ts
// todo_manager/store.ts

async function fetch(opts?: { force?: boolean }) {
    const [todosResp, convoResp] = await Promise.all([
        window.fetch('/api/todo-manager/todos'),
        window.fetch('/api/conversations'),
    ]);
    const todos: Todo[] = await todosResp.json();
    const conversations: ConversationWithPlugins[] = await convoResp.json();

    // Extract this plugin's conversations from the shared list.
    const myConvos = conversations.filter(c => 'todo-manager' in c.plugins);
    // ...
}
```

### Creating a conversation

1.  Frontend calls `POST /api/conversations` with `{id: cascadeId,
    workspaceId}`.
2.  Sidecar core inserts into `conversations`.
3.  Frontend (or plugin code) calls `POST /api/todo-manager/conversations` to
    attach the extension data: `{conversationId, todoId}`.

These are **two separate calls** — the sidecar core doesn't need to know about
the plugin's extension at creation time. If atomicity matters, the plugin
handler wraps both writes in a transaction.

--------------------------------------------------------------------------------

## 12. File Layout

```
sidecar/
    main.go                             # Boot: Open pool → plugin.Registry → register routes
    db/
        db.go                           # Open(ctx, dsn, plugins) *pgxpool.Pool
        schema.go                       # ApplySchema (advisory lock)
        schema.sql                      # Core schema: conversations table only
    store/
        conversation.go                 # Conversation, ConversationWithPlugins types
        conversation_repo.go            # ConversationRepo
        errors.go                       # ErrNotFound etc.
    plugin/
        plugin.go                       # Plugin interface, Registry
    todo_manager/
        schema.sql                      # todo_manager_todos + _conversation_meta
        types.go                        # Todo, ConversationMeta
        repo.go                         # TodoRepo + conv meta queries
        handler.go                      # HTTP handlers
        plugin.go                       # Implements plugin.Plugin
    BUILD
```

--------------------------------------------------------------------------------

## 13. What We Are NOT Building

| Omitted                       | Why                                         |
| ----------------------------- | ------------------------------------------- |
| `Database` interface wrapping | No backend swap; the interface costs more   |
: `pgxpool.Pool`                : than it gives                               :
| ORM (sqlx, gorm, ent)         | `pgx.CollectRows` + `pgx.RowToStructByName` |
:                               : is sufficient                               :
| `metadata JSONB` blob on      | Typed extension tables are safer and        |
: conversations                 : queryable                                   :
| Per workspace database        | Defeats the point — concurrency handled by  |
:                               : PostgreSQL                                  :
| Migration library             | Idempotent `CREATE IF NOT EXISTS` is enough |
:                               : while schema is young                       :
| SSE / push                    | Polling is fine for the data size and       |
:                               : refresh cadence                             :
| `mode` field on conversations | Transient UI concern — not a persistence    |
:                               : concern                                     :

--------------------------------------------------------------------------------

## 14. Still Open

| Question                           | Status                                  |
| ---------------------------------- | --------------------------------------- |
| How does the gateway identify the  | Needs a gateway design doc — options:   |
: workspace from an absolute `/api/` : `Referer` header parsing, session       :
: request?                           : cookie, `X-Workspace` header injected   :
:                                    : by the gateway before proxying          :
| `BatchGetConversationMeta` returns | Fine for now; can add generics later if |
: `map[string]any` — should it be    : needed                                  :
: typed?                             :                                         :
| Pagination on `GET                 | Add `?limit=&offset=` when the list     |
: /api/conversations`                : grows beyond a few hundred              :
