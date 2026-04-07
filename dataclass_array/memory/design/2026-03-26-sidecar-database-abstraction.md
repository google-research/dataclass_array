# Sidecar Database Abstraction

> **Status**: Design draft — 2026-03-26 (rev 2)
>
> **Scope**: Go sidecar backend + frontend data flow.
>
> **Starting point**: fresh — no migration from v1 JSON files.
>
> **First use-case**: TODO Manager plugin.

--------------------------------------------------------------------------------

## 1. Problem Statement

The v2 sidecar currently returns stub empty arrays. We need a real persistence
layer. The design constraints are:

-   **Multiple workspaces share one PostgreSQL database.** Each workspace has
    its own sidecar process, but they all connect to the same Postgres instance.
    Concurrency is a first-class concern, not an afterthought.
-   **Multiple plugins** will accumulate data over time. The storage abstraction
    must be clean enough to not become a mess as plugins multiply.
-   **Starting from scratch** — no v1 compatibility required.

--------------------------------------------------------------------------------

## 2. Goals and Non-Goals

**Goals**

-   Single shared PostgreSQL database for all workspaces.
-   Generic `Conversation` type owned by the sidecar core (not by any one
    plugin). Plugins attach plugin-specific data via extension tables.
-   Clean per-plugin repository layer — no SQL in HTTP handlers.
-   No wrapping of `pgx` — use `*pgxpool.Pool` directly.
-   Schema applied via embedded SQL at startup, with advisory-lock concurrency
    safety so multiple sidecars racing at boot are safe.
-   Workspace-scoped reads where appropriate — queries filter by `workspace_id`.
-   Frontend data access is explicit: REST endpoints, registered per plugin.

**Non-Goals**

-   No interface wrapping `pgxpool.Pool`.
-   No ORM, no generic CRUD base.
-   No real-time push (SSE/WebSocket) in v1 — polling is fine.
-   No v1 data import.

--------------------------------------------------------------------------------

## 3. Architecture Overview

```
┌── Browser (React) ──────────────────────────────────────────────────────┐
│  TodoStore: fetch("/api/todo-manager/todos")                            │
│             fetch("/api/todo-manager/conversations")                    │
└─────────────────────────────────────────────────────────────────────────┘
              │ HTTP (via Gateway prefix routing)
┌── Sidecar (Go) ──────────────────────────────────────────────────────────┐
│                                                                          │
│  main.go                                                                 │
│   └─ db.Open(dsn)  →  *pgxpool.Pool  (shared across all handlers)       │
│   └─ registerRoutes(mux, pool)                                           │
│        │                                                                 │
│        ├── /api/health            →  healthHandler                       │
│        ├── /api/conversations/*   →  ConversationHandler(pool)           │  ← sidecar core
│        └── /api/todo-manager/*    →  TodoManagerHandler(pool)            │  ← plugin
│                                                                          │
└──────────────────────────────────────┬───────────────────────────────────┘
                                       │ pgx pool (multiple connections)
┌── PostgreSQL (shared) ───────────────▼───────────────────────────────────┐
│  conversations      (sidecar-owned, generic)                             │
│  todo_manager_todos (plugin-owned)                                       │
│  todo_manager_conversation_meta  (join: conversations ↔ todo)            │
└──────────────────────────────────────────────────────────────────────────┘
                 ▲               ▲               ▲
           workspace A      workspace B      workspace C
           (sidecar proc)   (sidecar proc)   (sidecar proc)
```

--------------------------------------------------------------------------------

## 4. Do We Wrap `pgx`?

**Decision: No.**

`*pgxpool.Pool` is the concrete type threaded through the entire application. No
`Database` interface, no wrapper struct.

**Rationale:**

-   A `Database` interface over `pgx` requires re-implementing every primitive
    (`QueryRow`, `Exec`, `BeginTx`, batch...). Every new pgx feature forces a
    new interface method.
-   We will never swap Postgres for another backend — the interface buys
    nothing.
-   Cross-cutting concerns (logging, tracing) belong in a `pgx.QueryTracer`
    attached to the pool config — pgx's own extension point.
-   Tests use a real Postgres connection (via `pgxmock` or a test container).

The `db` package is a thin boot helper:

```go
// sidecar/db/db.go
package db

import (
    "context"
    "fmt"

    "github.com/jackc/pgx/v5/pgxpool"
)

// Open creates a connection pool and applies the embedded schema.
// The advisory lock inside applySchema makes this safe for concurrent callers.
func Open(ctx context.Context, dsn string) (*pgxpool.Pool, error) {
    pool, err := pgxpool.New(ctx, dsn)
    if err != nil {
        return nil, fmt.Errorf("open pool: %w", err)
    }
    if err := applySchema(ctx, pool); err != nil {
        pool.Close()
        return nil, fmt.Errorf("apply schema: %w", err)
    }
    return pool, nil
}
```

--------------------------------------------------------------------------------

## 5. Schema Strategy: No Migration Library

Since we're starting fresh and the schema will evolve rapidly, we use a **single
idempotent `CREATE TABLE IF NOT EXISTS` script** embedded in the binary, rather
than a versioned migration library. A migration library adds complexity (version
table, up/down files) that pays off when you need to evolve an existing
production schema — which we don't have yet.

When the schema genuinely needs an incompatible change, we drop and recreate the
database (during development) or add a new version field and handle it ad hoc.

```go
// sidecar/db/schema.go
package db

import (
    _ "embed"
    "context"
    "fmt"

    "github.com/jackc/pgx/v5/pgxpool"
)

//go:embed schema.sql
var schemaSQL string

// applySchema applies the schema idempotently.
// Uses a Postgres advisory lock so multiple sidecars racing at startup
// do not interleave DDL statements.
func applySchema(ctx context.Context, pool *pgxpool.Pool) error {
    conn, err := pool.Acquire(ctx)
    if err != nil {
        return fmt.Errorf("acquire conn for schema: %w", err)
    }
    defer conn.Release()

    // Advisory lock: all sidecars share lock key 0x616d /* 'am' */ schema.
    const lockKey = 0x616d736368656d61 // "amschema"
    if _, err := conn.Exec(ctx, "SELECT pg_advisory_lock($1)", lockKey); err != nil {
        return fmt.Errorf("advisory lock: %w", err)
    }
    defer conn.Exec(ctx, "SELECT pg_advisory_unlock($1)", lockKey) //nolint:errcheck

    if _, err := conn.Exec(ctx, schemaSQL); err != nil {
        return fmt.Errorf("apply schema.sql: %w", err)
    }
    return nil
}
```

`schema.sql` is a single file with `IF NOT EXISTS` guards throughout — running
it multiple times is safe.

--------------------------------------------------------------------------------

## 6. Generic Conversation (Sidecar Core)

A `Conversation` is a record that any plugin can create. It is the **sidecar's
own concept** — not tied to any one plugin's domain. Plugins attach their own
data by referencing the conversation ID in plugin-specific extension tables.

### 6.1 Go type

```go
// sidecar/store/conversation.go
package store

import "time"

// Conversation represents a Antigravity conversation launched from any plugin.
// It is the sidecar-owned generic record; plugin-specific data lives in
// extension tables that reference this ID.
type Conversation struct {
    ID             string    // UUID — primary key
    WorkspaceID    string    // which workspace created this conversation
    PluginID       string    // which plugin owns it (e.g. "todo-manager")
    ConversationID string    // Antigravity cascade ID (appears in URL as /c/<id>)
    LaunchedAt     time.Time
    Mode           string    // plugin-defined; e.g. "navigate" | "direct"
}
```

### 6.2 Repository

```go
// sidecar/store/conversation_repo.go
package store

import (
    "context"
    "fmt"
    "time"

    "github.com/jackc/pgx/v5"
    "github.com/jackc/pgx/v5/pgxpool"
)

type ConversationRepo struct {
    pool *pgxpool.Pool
}

func NewConversationRepo(pool *pgxpool.Pool) *ConversationRepo {
    return &ConversationRepo{pool: pool}
}

// Create inserts a new conversation and returns it with its assigned ID.
func (r *ConversationRepo) Create(ctx context.Context, workspaceID, pluginID, conversationID, mode string) (Conversation, error) {
    var c Conversation
    err := r.pool.QueryRow(ctx, `
        INSERT INTO conversations (workspace_id, plugin_id, conversation_id, mode)
        VALUES ($1, $2, $3, $4)
        RETURNING id, workspace_id, plugin_id, conversation_id, launched_at, mode`,
        workspaceID, pluginID, conversationID, mode,
    ).Scan(&c.ID, &c.WorkspaceID, &c.PluginID, &c.ConversationID, &c.LaunchedAt, &c.Mode)
    if err != nil {
        return Conversation{}, fmt.Errorf("insert conversation: %w", err)
    }
    return c, nil
}

// ListByPlugin returns all conversations for a given plugin, across all workspaces.
func (r *ConversationRepo) ListByPlugin(ctx context.Context, pluginID string) ([]Conversation, error) {
    rows, err := r.pool.Query(ctx, `
        SELECT id, workspace_id, plugin_id, conversation_id, launched_at, mode
        FROM conversations
        WHERE plugin_id = $1
        ORDER BY launched_at DESC`,
        pluginID)
    if err != nil {
        return nil, fmt.Errorf("list conversations: %w", err)
    }
    defer rows.Close()
    return pgx.CollectRows(rows, pgx.RowToStructByName[Conversation])
}
```

### 6.3 HTTP handler

The sidecar exposes a generic `/api/conversations/` endpoint so the frontend can
create and query conversations without knowing which plugin owns them.

```
GET  /api/conversations?plugin=todo-manager   → list conversations for plugin
POST /api/conversations                        → create (returns {id})
```

--------------------------------------------------------------------------------

## 7. Plugin-Specific Data: Extension Tables

Each plugin that needs to attach data to a conversation creates an **extension
table** with a FK to `conversations(id)`. No shared `metadata JSONB` blob — that
trades type safety for flexibility we don't need.

### 7.1 TODO Manager example

The `todo-manager` plugin links each conversation to a specific `Todo` and
records a `mode`:

```sql
-- In schema.sql

-- ── SIDECAR CORE ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversations (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id    TEXT        NOT NULL,
    plugin_id       TEXT        NOT NULL,
    conversation_id TEXT        NOT NULL,
    launched_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    mode            TEXT        NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS conversations_plugin_idx ON conversations(plugin_id);
CREATE INDEX IF NOT EXISTS conversations_workspace_idx ON conversations(workspace_id);

-- ── TODO MANAGER PLUGIN ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS todo_manager_todos (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    title      TEXT        NOT NULL,
    parent_id  UUID        REFERENCES todo_manager_todos(id) ON DELETE CASCADE,
    status     TEXT        NOT NULL DEFAULT 'not_started',
    content    TEXT        NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Extension table: attaches a todo to any sidecar conversation.
CREATE TABLE IF NOT EXISTS todo_manager_conversation_meta (
    conversation_id UUID PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    todo_id         UUID NOT NULL REFERENCES todo_manager_todos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS todo_conv_meta_todo_idx ON todo_manager_conversation_meta(todo_id);
```

### 7.2 Assembled Go type served to the frontend

The plugin repository **joins** the generic conversation with its own extension
table, producing a rich type:

```go
// sidecar/todo_manager/types.go
package todomanager

import (
    "time"
    "sidecar/store"
)

// TodoConversation is the assembled view served to the frontend.
// It embeds the generic Conversation and adds todo-specific fields.
type TodoConversation struct {
    store.Conversation             // id, workspace_id, launched_at, mode, ...
    TodoID         string          // from todo_manager_conversation_meta
    Todo           *Todo           // optionally JOIN-populated
}

type Todo struct {
    ID        string    `db:"id"`
    Title     string    `db:"title"`
    ParentID  *string   `db:"parent_id"`
    Status    string    `db:"status"`
    Content   string    `db:"content"`
    CreatedAt time.Time `db:"created_at"`
    UpdatedAt time.Time `db:"updated_at"`
    Children  []Todo    `db:"-"` // assembled in-process
}
```

The repository query:

```go
// Joins conversations + meta + todos in one query.
func (r *Repo) ListConversations(ctx context.Context) ([]TodoConversation, error) {
    rows, err := r.pool.Query(ctx, `
        SELECT
            c.id, c.workspace_id, c.plugin_id, c.conversation_id, c.launched_at, c.mode,
            m.todo_id,
            t.title AS todo_title, t.status AS todo_status
        FROM conversations c
        JOIN todo_manager_conversation_meta m ON m.conversation_id = c.id
        JOIN todo_manager_todos t ON t.id = m.todo_id
        WHERE c.plugin_id = 'todo-manager'
        ORDER BY c.launched_at DESC`)
    ...
}
```

**Creating a TodoConversation** is one atomic transaction — it inserts into both
`conversations` and `todo_manager_conversation_meta`:

```go
func (r *Repo) CreateConversation(ctx context.Context, req CreateConversationReq) (TodoConversation, error) {
    tx, err := r.pool.Begin(ctx)
    if err != nil {
        return TodoConversation{}, err
    }
    defer tx.Rollback(ctx)

    var c store.Conversation
    err = tx.QueryRow(ctx, `
        INSERT INTO conversations (workspace_id, plugin_id, conversation_id, mode)
        VALUES ($1, 'todo-manager', $2, $3)
        RETURNING id, workspace_id, plugin_id, conversation_id, launched_at, mode`,
        req.WorkspaceID, req.ConversationID, req.Mode,
    ).Scan(&c.ID, &c.WorkspaceID, &c.PluginID, &c.ConversationID, &c.LaunchedAt, &c.Mode)
    if err != nil {
        return TodoConversation{}, fmt.Errorf("insert conversation: %w", err)
    }

    if _, err = tx.Exec(ctx, `
        INSERT INTO todo_manager_conversation_meta (conversation_id, todo_id)
        VALUES ($1, $2)`, c.ID, req.TodoID); err != nil {
        return TodoConversation{}, fmt.Errorf("insert conversation meta: %w", err)
    }

    if err = tx.Commit(ctx); err != nil {
        return TodoConversation{}, fmt.Errorf("commit: %w", err)
    }
    return TodoConversation{Conversation: c, TodoID: req.TodoID}, nil
}
```

--------------------------------------------------------------------------------

## 8. Concurrency — Multiple Sidecars, One Database

**The hard constraint**: every workspace has its own sidecar process, and all
processes connect to the same Postgres. We can have N concurrent writers.

### What PostgreSQL gives us for free

-   **Atomic INSERT/UPDATE/DELETE** — no application-level mutex needed for
    individual writes.
-   **SERIALIZABLE / READ COMMITTED** isolation — default `READ COMMITTED` is
    sufficient for our use cases (no read-then-write invariants on the same row
    from two clients simultaneously).
-   **UNIQUE constraints** — enforce uniqueness at the DB level, not in the app.

### Patterns we apply

| Situation                         | Mechanism                                |
| --------------------------------- | ---------------------------------------- |
| Schema DDL at startup (multiple   | `pg_advisory_lock` in `applySchema` —    |
: sidecars)                         : only one executes DDL, others wait       :
| Simple inserts (new todo, new     | Plain `INSERT` — each sidecar writes its |
: conversation)                     : own rows independently                   :
| Update a todo (optimistic)        | `UPDATE ... WHERE id=$1 AND              |
:                                   : updated_at=$2 RETURNING id` — returns 0  :
:                                   : rows if stale; caller retries            :
| Read-then-mutate (e.g. reparent a | `SELECT ... FOR UPDATE` inside an        |
: subtree)                          : explicit `BEGIN`/`COMMIT`                :
| Ensuring a conversation_id is not | `UNIQUE(conversation_id)` constraint on  |
: duplicated                        : `conversations` table                    :

### Workspace-scoped queries

All sidecar processes share the same `conversations` table. Where data is
workspace-specific (e.g. "show me my conversations"), queries **filter by
`workspace_id`**. The `workspace_id` is injected by the sidecar at startup from
the `--workspace` flag the gateway already passes.

Todos, however, are **shared across all workspaces** — the todo tree is a global
data model, not per-workspace. Only the conversations that link to those todos
are workspace-scoped.

--------------------------------------------------------------------------------

## 9. Frontend Data Flow

The React frontend (plugin code) calls the sidecar via relative `fetch()` calls
through the gateway. Each plugin owns its own URL namespace.

### URL namespacing

```
/api/conversations     ← sidecar core (generic)
/api/todo-manager/*    ← todo-manager plugin
/api/<plugin-id>/*     ← other future plugins
```

### Todo Manager endpoints

| Method   | Path                              | Description          |
| -------- | --------------------------------- | -------------------- |
| `GET`    | `/api/todo-manager/todos`         | Returns full todo    |
:          :                                   : tree                 :
| `POST`   | `/api/todo-manager/todos`         | Create a new todo    |
| `PUT`    | `/api/todo-manager/todos/:id`     | Update               |
:          :                                   : title/status/content :
| `DELETE` | `/api/todo-manager/todos/:id`     | Delete (cascades to  |
:          :                                   : conversation meta)   :
| `GET`    | `/api/todo-manager/conversations` | Returns              |
:          :                                   : `[]TodoConversation` :
:          :                                   : (joined)             :
| `POST`   | `/api/todo-manager/conversations` | Create               |
:          :                                   : conversation+meta    :
:          :                                   : atomically           :

### Generic conversation endpoints

Method | Path                             | Description
------ | -------------------------------- | -----------------------------------
`GET`  | `/api/conversations?plugin=<id>` | List all conversations for a plugin

### How the React store calls this

```ts
// todo_manager/store.ts

async function fetch(opts?: { force?: boolean }) {
    const [todosResp, convoResp] = await Promise.all([
        window.fetch('api/todo-manager/todos'),
        window.fetch('api/todo-manager/conversations'),
    ]);
    // ...
}

async function registerConversation(entry: Omit<TodoConversation, 'id'>) {
    await window.fetch('api/todo-manager/conversations', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(entry),
    });
}
```

Note: `fetch('api/...')` not `fetch('/api/...')` — relative paths are required
by the gateway's root isolation middleware. The sidecar receives the path
already stripped of the workspace prefix.

### Polling strategy

The frontend polls on mount and on explicit user refresh. The sidecar does NOT
push — no SSE, no WebSocket. `ConversationStatus` (running/idle) is derived from
`useAgentStateProvider` (Antigravity's own stream), not from the database.

--------------------------------------------------------------------------------

## 10. File Layout

```
sidecar/
    main.go                         # Startup: Open pool, register routes
    db/
        db.go                       # Open(ctx, dsn) *pgxpool.Pool
        schema.go                   # applySchema (advisory lock + embedded SQL)
        schema.sql                  # Single idempotent CREATE TABLE IF NOT EXISTS file
    store/
        conversation.go             # Conversation type
        conversation_repo.go        # ConversationRepo (sidecar core)
    todo_manager/
        types.go                    # Todo, TodoConversation, CreateConversationReq
        todo_repo.go                # TodoRepo (CRUD on todo_manager_todos)
        conversation_repo.go        # Repo.ListConversations, Repo.CreateConversation
        handler.go                  # HTTP handlers — call repos, no SQL here
    BUILD
```

--------------------------------------------------------------------------------

## 11. What We Are NOT Building

| Omitted                           | Why                                      |
| --------------------------------- | ---------------------------------------- |
| `Database` interface wrapping     | No backend swap planned; interfacing pgx |
: `pgxpool.Pool`                    : costs more than it gives                 :
| ORM (sqlx, gorm, ent)             | `pgx.CollectRows` +                      |
:                                   : `pgx.RowToStructByName` is sufficient    :
| Generic CRUD base struct          | Plugin data models are distinct enough   |
| `metadata JSONB` on conversations | Typed extension tables are safer and     |
:                                   : queryable                                :
| Per-workspace database isolation  | Defeats the purpose of sharing state;    |
:                                   : concurrency via PostgreSQL               :
| Migration library                 | Idempotent `CREATE IF NOT EXISTS` is     |
:                                   : enough while schema is young             :
| SSE / push                        | Polling is fine for the current data     |
:                                   : size and refresh rate                    :

--------------------------------------------------------------------------------

## 12. Open Questions

| Question                            | Status                                 |
| ----------------------------------- | -------------------------------------- |
| Where does the shared Postgres run? | To decide — could be system Postgres   |
: (socket path, port, user)           : on a fixed port, or a Unix socket at   :
:                                     : `~/.agent-manager/pgsql/.s.PGSQL.5432` :
:                                     : owned by the user.                     :
| Who starts Postgres if it's not     | The sidecar or a one-time CLI setup    |
: already running?                    : command? Recommendation\: CLI          :
:                                     : (`agent_manager.py setup`) to keep     :
:                                     : sidecar startup simple.                :
| `--workspace` flag format — how is  | Gateway already passes it; verify flag |
: it passed today?                    : name in `main.go`.                     :
| Notion todos: stored in DB or       | Fetched fresh from Notion API on every |
: fetched fresh?                      : `GET /api/todo-manager/todos` with TTL :
:                                     : cache, NOT stored. Only native todos   :
:                                     : are DB-backed.                         :
