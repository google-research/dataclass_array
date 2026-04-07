# Sidecar Database Abstraction

> **Status**: Design draft — 2026-03-27 (rev 4) **Previous**:
> `2026-03-26-sidecar-database-abstraction-v3.md`

--------------------------------------------------------------------------------

## 1. Problem Statement

The v2 sidecar returns stub empty arrays. We need a real persistence layer.
Constraints:

-   **All workspaces share one PostgreSQL database.** Each workspace has its own
    sidecar process, but all connect to the same Postgres. Concurrency is a
    first-class concern.
-   **`Conversation` is a sidecar core concept.** Any plugin can annotate a
    conversation with plugin-specific metadata. No plugin owns it.
-   **Plugins are hardcoded** — there is no runtime plugin discovery. The plugin
    interface exists only where polymorphism pays for itself.

--------------------------------------------------------------------------------

## 2. Plugin Registration: Hardcoded, Not Dynamic

### The question

The v3 design had a full `Plugin` interface: `ID()`, `SchemaSQL()`,
`RegisterRoutes()`, `GetConversationMeta()`. The user's question was: **does
this pay for itself, or would hardcoding be cleaner?**

### Decision: intermediate approach

Three options were considered:

| Option             | How it works                      | Verdict            |
| ------------------ | --------------------------------- | ------------------ |
| **A — Full dynamic | All plugin concerns behind one    | Over-abstracted.   |
: interface**        : interface; registered via a slice : Registry loop for  :
:                    :                                   : schema/routes adds :
:                    :                                   : no value when      :
:                    :                                   : plugins are        :
:                    :                                   : hardcoded          :
| **B — Fully        | `main.go` calls                   | Simple but leaves  |
: hardcoded**        : `todomanager.ApplySchema(...)`,   : no structure for   :
:                    : `todomanager.RegisterRoutes(...)` : the one case where :
:                    : directly                          : polymorphism is    :
:                    :                                   : real               :
| **C — Minimal      | Schema + routes wired explicitly  | Best of both       |
: interface,         : in `main.go`; interface used only :                    :
: explicit wiring**  : for conversation meta assembly    :                    :

**We use Option C.** The interface is scoped to the one operation that is
genuinely polymorphic: assembling plugin annotations when serving a
conversation. Everything else (schema, routes) is explicit:

```go
// sidecar/main.go

func main() {
    pool, _ := db.Open(ctx, *dbDSN)

    // ── Schema — explicit, each plugin owns its own idempotent SQL ──────
    db.ApplyCoreSchema(ctx, pool)
    todomanager.ApplySchema(ctx, pool)
    // future plugins added here

    mux := http.NewServeMux()
    mux.HandleFunc("/api/health", handleHealth)

    // ── Routes — explicit ────────────────────────────────────────────────
    todoPlugin := todomanager.New(pool)
    todoPlugin.RegisterRoutes(mux)
    // future plugins added here

    // ── Conversation handler — polymorphic plugin list ───────────────────
    // Only the conversation handler needs to iterate plugins at runtime.
    plugins := []plugin.ConversationPlugin{todoPlugin}
    mux.Handle("/api/conversations", newConversationHandler(pool, plugins))
    mux.Handle("/api/conversations/", newConversationHandler(pool, plugins))

    http.ListenAndServe(fmt.Sprintf(":%d", *port), mux)
}
```

### The minimal interface

```go
// sidecar/plugin/plugin.go
package plugin

import "context"

// ConversationPlugin is the only interface plugins must implement.
// It exists solely to support the conversation metadata assembly loop
// in GET /api/conversations and GET /api/conversations/:id.
//
// Schema application and HTTP route registration are handled explicitly
// in main.go — they do not belong to an interface.
type ConversationPlugin interface {
    // PluginID returns the plugin's unique key (e.g. "todo-manager").
    // Used as the key in the response's "plugins" map.
    PluginID() string

    // BatchGetConversationMeta returns plugin metadata for the given
    // conversation IDs. Missing IDs are omitted from the map.
    // Called once per LIST request, once per GET request.
    BatchGetConversationMeta(ctx context.Context, ids []string) (map[string]any, error)
}
```

This is the entire interface. `SchemaSQL() string` is gone — it was only needed
by the registry loop, which is now explicit in `main.go`.

--------------------------------------------------------------------------------

## 3. Core: `Conversation`

### 3.1 Single struct

The v3 design had two types: `Conversation` and `ConversationWithPlugins`. The
user asked why. **There is no good reason** — collapse them into one:

```go
// sidecar/store/conversation.go
package store

import "time"

// Conversation is the sidecar's core record for any Antigravity conversation.
// Plugin-specific data is stored in plugin extension tables and optionally
// included here in the Plugins map when serving API responses.
type Conversation struct {
    ID          string         `db:"id" json:"id"`           // Antigravity cascade ID
    WorkspaceID string         `db:"workspace_id" json:"workspaceId"`
    CreatedAt   time.Time      `db:"created_at" json:"createdAt"`
    Plugins     map[string]any `db:"-" json:"plugins,omitempty"` // assembled at query time
}
```

`Plugins` is `nil` when the struct is used internally (e.g. in the repo before
annotation). It is always populated before serialising to JSON.

### 3.2 Core schema

```sql
-- sidecar/db/core_schema.sql

CREATE TABLE IF NOT EXISTS conversations (
    id           TEXT        PRIMARY KEY,   -- Antigravity cascade ID (/c/<id>)
    workspace_id TEXT        NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_workspace_idx ON conversations(workspace_id);
```

### 3.3 Repository

```go
// sidecar/store/conversation_repo.go

type ConversationRepo struct{ pool *pgxpool.Pool }

// Create inserts a new conversation. Returns ErrConflict if the ID already
// exists — a duplicate cascade ID is a bug, not a routine race.
func (r *ConversationRepo) Create(ctx context.Context, id, workspaceID string) (Conversation, error) {
    var c Conversation
    err := r.pool.QueryRow(ctx,
        `INSERT INTO conversations (id, workspace_id)
         VALUES ($1, $2)
         RETURNING id, workspace_id, created_at`,
        id, workspaceID,
    ).Scan(&c.ID, &c.WorkspaceID, &c.CreatedAt)
    if isUniqueViolation(err) {
        return Conversation{}, fmt.Errorf("conversation %s already exists: %w", id, ErrConflict)
    }
    return c, err
}

func (r *ConversationRepo) ListAll(ctx context.Context) ([]Conversation, error) { ... }
func (r *ConversationRepo) Get(ctx context.Context, id string) (Conversation, error) { ... }
```

Duplicate cascade IDs returning `ErrConflict` (HTTP 409) surfaces bugs instead
of swallowing them.

### 3.4 Handler — assembling the plugin map

Plugin metadata is fetched in **batch per plugin** (not per conversation) to
avoid N+1:

```go
func (h *conversationHandler) assemblePlugins(ctx context.Context, convos []Conversation) error {
    ids := convIDs(convos)
    byID := make(map[string]*Conversation, len(convos))
    for i := range convos { byID[convos[i].ID] = &convos[i] }

    for _, p := range h.plugins {
        metas, err := p.BatchGetConversationMeta(ctx, ids)
        if err != nil {
            return fmt.Errorf("plugin %s: %w", p.PluginID(), err)
        }
        for convID, meta := range metas {
            c := byID[convID]
            if c.Plugins == nil { c.Plugins = map[string]any{} }
            c.Plugins[p.PluginID()] = meta
        }
    }
    return nil
}
```

--------------------------------------------------------------------------------

## 4. Plugin-Specific Data: Extension Tables

Each plugin's schema SQL lives in the **plugin's own folder** and is applied
explicitly from `main.go`. No interface method needed.

### 4.1 `todo_attempt` table (renamed from `todo_manager_conversation_meta`)

The table is renamed `todo_attempt` because it captures the concept of a
**single attempt at completing a todo** — it is not just metadata.

A `todo_attempt` also gets its own `attempt_status` to record the lifecycle of
that specific attempt, independent of Antigravity's agent state stream (which is
ephemeral).

```sql
-- sidecar/todo_manager/schema.sql

-- ── Todo status ──────────────────────────────────────────────────────────
--
-- Stored as TEXT, validated in Go. No DB ENUM or CHECK constraint yet.
-- Rationale: status values are still evolving. A CHECK constraint is easy
-- to add later once values stabilize; an ENUM is harder to ALTER.
-- See §4.2 for the tradeoff discussion.

CREATE TABLE IF NOT EXISTS todo_manager_todos (
    id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    title      TEXT        NOT NULL,
    parent_id  UUID        REFERENCES todo_manager_todos(id) ON DELETE CASCADE,
    status     TEXT        NOT NULL DEFAULT 'not_started',
    content    TEXT        NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── todo_attempt ─────────────────────────────────────────────────────────
--
-- Each row = one attempt to complete a todo via a Antigravity conversation.
-- A todo can have many attempts (multiple tries). A conversation currently
-- maps to at most one attempt, but this is enforced by a UNIQUE constraint
-- that can be dropped later if many-to-many is needed.

CREATE TABLE IF NOT EXISTS todo_attempt (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    todo_id         UUID        NOT NULL REFERENCES todo_manager_todos(id) ON DELETE CASCADE,
    conversation_id TEXT        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    attempt_status  TEXT        NOT NULL DEFAULT 'pending',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (conversation_id)  -- one conversation → at most one attempt (relaxable later)
);

CREATE INDEX IF NOT EXISTS todo_attempt_todo_idx ON todo_attempt(todo_id);
```

### 4.2 `attempt_status` values

Value       | Meaning
----------- | -------------------------------------------
`pending`   | Conversation created, agent not yet started
`running`   | Agent actively working
`succeeded` | Agent completed the task
`failed`    | Agent errored or was stopped mid-task
`abandoned` | User manually dismissed this attempt

### 4.3 `todo_status` vs `attempt_status` — ENUM or TEXT?

**The tradeoff:**

| Approach        | Adding a value     | Removing a value | Type safety       |
| --------------- | ------------------ | ---------------- | ----------------- |
| PostgreSQL      | `ALTER TYPE ADD    | Impossible       | DB-enforced       |
: `ENUM`          : VALUE` —           : without          :                   :
:                 : non-transactional, : recreating       :                   :
:                 : can't roll back    :                  :                   :
| Lookup table    | `INSERT INTO       | `DELETE FROM     | FK-enforced       |
:                 : statuses VALUES    : statuses VALUES  :                   :
:                 : (...)`             : (...)`           :                   :
| `TEXT` + CHECK  | `ALTER TABLE DROP  | Same             | DB-enforced       |
: constraint      : CONSTRAINT + ADD   :                  :                   :
:                 : CONSTRAINT`        :                  :                   :
| `TEXT` + Go     | Edit Go code       | Edit Go code     | App-enforced only |
: validation only :                    :                  :                   :

**Decision for now: `TEXT` validated in Go only.**

The status values for both `todo_status` and `attempt_status` are still evolving
rapidly. During this phase, enforcing validity in the application layer (a Go
constant set) is sufficient and avoids painful DDL churn.

**Planned upgrade path:** once the values stabilize, add a PostgreSQL CHECK
constraint. If performance or data integrity requirements grow, move to a lookup
table. ENUM is not recommended because removing values is impractical.

### 4.4 Go types

```go
// sidecar/todo_manager/types.go
package todomanager

import "time"

// TodoStatus enumerates valid todo status values.
// Validated in Go; stored as TEXT in the database.
type TodoStatus string

const (
    TodoStatusNotStarted TodoStatus = "not_started"
    TodoStatusInProgress TodoStatus = "in_progress"
    TodoStatusNeedsReview TodoStatus = "needs_review"
    TodoStatusDone        TodoStatus = "done"
)

// AttemptStatus enumerates valid attempt status values.
type AttemptStatus string

const (
    AttemptStatusPending   AttemptStatus = "pending"
    AttemptStatusRunning   AttemptStatus = "running"
    AttemptStatusSucceeded AttemptStatus = "succeeded"
    AttemptStatusFailed    AttemptStatus = "failed"
    AttemptStatusAbandoned AttemptStatus = "abandoned"
)

type Todo struct {
    ID        string     `db:"id"`
    Title     string     `db:"title"`
    ParentID  *string    `db:"parent_id"`
    Status    TodoStatus `db:"status"`
    Content   string     `db:"content"`
    CreatedAt time.Time  `db:"created_at"`
    UpdatedAt time.Time  `db:"updated_at"`
    Children  []Todo     `db:"-"` // assembled in-process from flat list
}

// TodoAttempt is a single attempt at completing a todo.
type TodoAttempt struct {
    ID             string        `db:"id"`
    TodoID         string        `db:"todo_id"`
    ConversationID string        `db:"conversation_id"`
    Status         AttemptStatus `db:"attempt_status"`
    CreatedAt      time.Time     `db:"created_at"`
    UpdatedAt      time.Time     `db:"updated_at"`
    Todo           *Todo         `db:"-"` // optionally populated
}

// AttemptMeta is the plugin's contribution to the conversation's plugins map.
// Appears as: {"todo-manager": AttemptMeta}.
type AttemptMeta struct {
    AttemptID     string        `json:"attemptId"`
    TodoID        string        `json:"todoId"`
    AttemptStatus AttemptStatus `json:"attemptStatus"`
    Todo          *Todo         `json:"todo,omitempty"`
}
```

### 4.5 Plugin struct (implements `ConversationPlugin`)

```go
// sidecar/todo_manager/plugin.go
package todomanager

type Plugin struct { repo *Repo }

func New(pool *pgxpool.Pool) *Plugin { return &Plugin{repo: NewRepo(pool)} }

func (p *Plugin) PluginID() string { return "todo-manager" }

func (p *Plugin) RegisterRoutes(mux *http.ServeMux) {
    h := newHandler(p.repo)
    mux.HandleFunc("/api/todo-manager/todos", h.handleTodos)
    mux.HandleFunc("/api/todo-manager/todos/", h.handleTodo)
    mux.HandleFunc("/api/todo-manager/attempts", h.handleAttempts)
    mux.HandleFunc("/api/todo-manager/attempts/", h.handleAttempt)
}

func (p *Plugin) BatchGetConversationMeta(ctx context.Context, ids []string) (map[string]any, error) {
    attempts, err := p.repo.BatchGetAttemptsByConversation(ctx, ids)
    if err != nil {
        return nil, err
    }
    result := make(map[string]any, len(attempts))
    for convID, attempt := range attempts {
        result[convID] = AttemptMeta{
            AttemptID:     attempt.ID,
            TodoID:        attempt.TodoID,
            AttemptStatus: attempt.Status,
            Todo:          attempt.Todo,
        }
    }
    return result, nil
}
```

--------------------------------------------------------------------------------

## 5. Concurrency: Multiple Sidecars, One Database

| Situation                           | Mechanism                              |
| ----------------------------------- | -------------------------------------- |
| Schema DDL at startup (N sidecars   | `pg_advisory_lock` — one executes DDL, |
: racing)                             : others wait, then no-op on `IF NOT     :
:                                     : EXISTS`                                :
| Inserting a new conversation        | Plain `INSERT` — fail with             |
:                                     : `ErrConflict` (409) on duplicate       :
:                                     : cascade ID                             :
| Creating a todo                     | Plain `INSERT` — no conflict possible  |
:                                     : across sidecars                        :
| Updating a todo (optimistic)        | `UPDATE ... WHERE id=$1 AND            |
:                                     : updated_at=$2` — 0 rows → 409          :
| Updating attempt_status             | `UPDATE todo_attempt SET               |
:                                     : attempt_status=$1, updated_at=now()    :
:                                     : WHERE id=$2` — last writer wins        :
:                                     : (status updates are monotonic)         :
| Read-then-mutate (reparent subtree) | `SELECT ... FOR UPDATE` + explicit     |
:                                     : transaction                            :

--------------------------------------------------------------------------------

## 6. API

### 6.1 Core conversation endpoints

```
POST /api/conversations           body: {id, workspaceId}
                                  → 201 {id, workspaceId, createdAt, plugins: {}}
                                  → 409 if id already exists

GET  /api/conversations           → [{id, workspaceId, createdAt, plugins: {...}}]
GET  /api/conversations/:id       → {id, workspaceId, createdAt, plugins: {...}}
```

### 6.2 Todo manager endpoints

```
GET    /api/todo-manager/todos           → todo tree (children assembled in-process)
POST   /api/todo-manager/todos           → create {title, parentId?}
PUT    /api/todo-manager/todos/:id       → update {title?, status?, content?}
DELETE /api/todo-manager/todos/:id       → delete (cascades to attempts)

POST   /api/todo-manager/attempts        → create {todoId, conversationId}
GET    /api/todo-manager/attempts?todoId=  → list attempts for a todo
PUT    /api/todo-manager/attempts/:id    → update {attemptStatus}
```

The `POST /api/todo-manager/attempts` handler wraps both the conversation insert
(if the conversation doesn't exist yet) and the `todo_attempt` insert in one
transaction.

### 6.3 Response shape

```ts
interface Conversation {
    id: string;          // Antigravity cascade ID
    workspaceId: string;
    createdAt: string;
    plugins: {
        "todo-manager"?: {
            attemptId: string;
            todoId: string;
            attemptStatus: "pending" | "running" | "succeeded" | "failed" | "abandoned";
            todo?: { id: string; title: string; status: string; ... };
        };
    };
}
```

--------------------------------------------------------------------------------

## 7. Postgres Lifecycle

**The Gateway owns Postgres** — same supervision pattern as sidecar and LS.

```
Gateway startup:
  1. Start/attach Postgres
  2. Wait for connection
  3. Start sidecar  (receives --db_dsn flag, opens pool, applies schema)
  4. Start LS
```

--------------------------------------------------------------------------------

## 8. File Layout

```
sidecar/
    main.go                         # Explicit wiring: schema, routes, plugin list
    db/
        db.go                       # Open(ctx, dsn) *pgxpool.Pool
        schema.go                   # ApplyCoreSchema (advisory lock + embedded SQL)
        core_schema.sql             # conversations table only
    store/
        conversation.go             # Conversation type (with Plugins field)
        conversation_repo.go        # ConversationRepo
        errors.go                   # ErrNotFound, ErrConflict
    plugin/
        plugin.go                   # ConversationPlugin interface only
    todo_manager/
        schema.sql                  # todo_manager_todos + todo_attempt
        schema.go                   # ApplySchema(ctx, pool) — called from main
        types.go                    # Todo, TodoAttempt, AttemptMeta, status consts
        repo.go                     # TodoRepo + AttemptRepo
        handler.go                  # HTTP handlers
        plugin.go                   # Implements plugin.ConversationPlugin
    BUILD
```

--------------------------------------------------------------------------------

## 9. What We Are NOT Building

| Omitted                          | Why                                      |
| -------------------------------- | ---------------------------------------- |
| Dynamic plugin discovery         | Plugins are hardcoded — the interface is |
:                                  : scoped to conversation meta only         :
| `SchemaSQL() string` on Plugin   | Schema applied explicitly; no need for   |
: interface                        : interface polymorphism here              :
| `ConversationWithPlugins` second | Merged into `Conversation.Plugins`       |
: struct                           :                                          :
| `ON CONFLICT DO NOTHING` on      | Conflict is a bug → explicit 409 error   |
: conversations                    :                                          :
| PostgreSQL ENUM for status       | Too brittle during rapid iteration; TEXT |
:                                  : + Go constants for now                   :
| `metadata JSONB` blob            | Typed extension tables are queryable and |
:                                  : safe                                     :
| Per-workspace database           | One DB, concurrency via PostgreSQL       |
| Migration library                | Idempotent `IF NOT EXISTS` sufficient    |
:                                  : while schema is young                    :

--------------------------------------------------------------------------------

## 10. Still Open

| Question                            | Notes                                  |
| ----------------------------------- | -------------------------------------- |
| How does the gateway route absolute | Gateway design doc TBD — options:      |
: `/api/` requests to the correct     : Referer header, injected `X-Workspace` :
: workspace's sidecar?                : header, session cookie                 :
| When does `attempt_status` get      | Sidecar polls agent state? Frontend    |
: updated?                            : PATCHes it? Needs decision             :
| Pagination on `GET                  | Add `?limit=&offset=` when list grows  |
: /api/conversations`                 :                                        :
| CHECK constraint threshold          | Add once both status enums have been   |
:                                     : stable for ≥2 weeks                    :
