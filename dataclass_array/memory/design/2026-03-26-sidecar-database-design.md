# Sidecar Database Abstraction Design

**Date:** 2026-03-26 **Status:** Draft

--------------------------------------------------------------------------------

## 1. Problem statement

The v1 sidecar stored plugin data (e.g. `TodoConversation` records) in flat JSON
files (`~/.agent-manager/conversations.json`). This approach does not scale:

-   No concurrent write safety (two sidecar instances race on the same file).
-   No indexing or querying — every read loads the entire file.
-   Schema evolution requires hand-rolled migration code.

The v2 sidecar will use **PostgreSQL** (via the
[pgx](https://github.com/jackc/pgx) Go driver) as its database.

This document designs the layered abstraction so that:

1.  The sidecar manages one PostgreSQL connection pool.
2.  Each plugin receives a domain-specific repository, not a raw DB handle.
3.  SQL is encapsulated; plugin business logic never sees a query string.

--------------------------------------------------------------------------------

## 2. Do we actually need these abstractions?

### 2.1 Is a PostgreSQL wrapper layer necessary?

**Short answer: No generic "DB abstraction" — but yes, a testable interface.**

pgx already provides an excellent, idiomatic Go API. Wrapping it in a `Database`
interface that hides `pgxpool.Pool` would be pointless indirection: we are not
going to swap PostgreSQL for SQLite, and pgx is not a legacy API that needs
cleaning up.

What *is* worth having is a thin interface over `pgxpool.Pool` so that tests can
inject a fake or a `pgxmock` without starting a real database. The standard Go
pattern is to accept an interface (`db.Querier`) rather than a concrete type.
pgx v5 ships its own `pgxpool.Pool` but exposes the `pgx.Tx` and `pgconn`
interfaces; we can carve out whatever subset we need.

**Decision:** expose a `Querier` interface (`ExecContext`, `QueryRow`,
`QueryRows`) that `*pgxpool.Pool` and `*pgx.Tx` both satisfy. The sidecar
creates and owns the concrete pool. Plugins never see `pgxpool.Pool` — only the
`Querier` passed to their repository constructor.

### 2.2 Is a per-plugin domain mapping layer necessary?

**Yes — and for a principled reason, not ceremony.**

Consider the alternative: plugin handlers call `pool.Query(ctx, "SELECT ... FROM
conversations WHERE todo_id = $1", todoId)` directly. Problems:

-   SQL leaks into HTTP handler business logic, mixing abstraction levels.
-   The same query is copy–pasted across handlers.
-   Tests must mock at the SQL level, not the domain level.
-   Schema changes require hunting every query string in the codebase.

The **repository pattern** solves this cleanly: each plugin defines a
`XxxRepository` interface with domain-level methods (`GetConversations(todoId
string) ([]TodoConversation, error)`), and one concrete implementation backed by
pgx. HTTP handlers operate on the interface; tests substitute a fake.

This is not a generic ORM — there is no reflection, no `struct` tag scanning, no
`Model()` helper. Each repository is a small, hand-written adapter (~50–100
lines) that translates between SQL rows and domain structs.

--------------------------------------------------------------------------------

## 3. Architecture

```
sidecar main.go
    │
    ├── pgxpool.Pool  (created once at startup, closed at shutdown)
    │       │
    │       └── wraps pgx connection pool to Postgres
    │
    ├── db.Querier  (interface, satisfied by *pgxpool.Pool and *pgx.Tx)
    │       │
    │       ├── todo-manager plugin
    │       │       └── todo.Repository  (interface)
    │       │               └── todo.pgxRepository  (concrete, holds Querier)
    │       │                       SQL ↔ TodoConversation mapping
    │       │
    │       └── future-plugin
    │               └── future.Repository  ...
    │
    └── HTTP handlers  (receive repository interfaces as dependencies)
```

### Dependency flow

```
main.go
  pool := pgxpool.New(ctx, dsn)
  q    := db.NewQuerier(pool)        // wraps pool, no new layer
  repo := todo.NewRepository(q)      // domain-level adapter
  srv  := todo.NewServer(repo)       // HTTP handlers
  mux.Handle("/api/conversations", srv)
```

No globals. Each dependency is passed explicitly down the call chain (api-design
principle: *pass dependencies, don't reach for globals*).

--------------------------------------------------------------------------------

## 4. Layer specifications

### 4.1 `sidecar/db` — Querier interface

```go
// Package db defines the minimal database interface used by sidecar plugins.
// It is satisfied by *pgxpool.Pool, *pgx.Tx, and any test double.
package db

import (
    "context"

    "github.com/jackc/pgx/v5"
    "github.com/jackc/pgx/v5/pgconn"
)

// Querier is the subset of pgxpool.Pool that plugins need.
// Both *pgxpool.Pool and *pgx.Tx implement this interface.
type Querier interface {
    Exec(ctx context.Context, sql string, args ...any) (pgconn.CommandTag, error)
    Query(ctx context.Context, sql string, args ...any) (pgx.Rows, error)
    QueryRow(ctx context.Context, sql string, args ...any) pgx.Row
}

// WithTx runs fn inside a transaction, committing on success and rolling back
// on error or panic. pool must be *pgxpool.Pool.
func WithTx(ctx context.Context, pool interface {
    Begin(context.Context) (pgx.Tx, error)
}, fn func(Querier) error) error {
    tx, err := pool.Begin(ctx)
    if err != nil {
        return err
    }
    defer func() {
        if p := recover(); p != nil {
            _ = tx.Rollback(ctx)
            panic(p)
        }
    }()
    if err := fn(tx); err != nil {
        _ = tx.Rollback(ctx)
        return err
    }
    return tx.Commit(ctx)
}
```

**Why only three methods?** YAGNI. Add `CopyFrom`, `SendBatch`, etc. when a
plugin actually needs them. The interface stays narrow.

**Why not pgxpool.Pool directly?** Tests would need a real PostgreSQL process.
With `Querier`, a test can pass a `pgxmock` or a simple in-memory fake.

### 4.2 `sidecar/db/migrate` — Schema migrations

Lightweight, no framework. Migrations are numbered SQL files embedded in the
binary via `go:embed`:

```
sidecar/db/migrate/
    0001_create_conversations.sql
    0002_add_template_col.sql
    ...
```

`migrate.Run(ctx, pool)` applies any unapplied migrations in order, using a
`schema_migrations` table as the applied-set tracker. This is the minimal
version of what Flyway / golang-migrate do, without the external dependency.

```go
// migrate.AppliedMigrations returns the set of already-applied migration IDs.
// migrate.Apply runs a single migration in a transaction.
// migrate.Run applies all pending migrations on startup.
```

### 4.3 Plugin repository — example: `sidecar/todo`

The TODO plugin needs to store `TodoConversation` records (see v2 research doc).

```go
// Package todo implements the sidecar-side of the TODO Manager plugin.
package todo

import (
    "context"
    "time"

    ".../sidecar/db"
)

// Conversation is the domain object (mirrors the frontend's TodoConversation).
type Conversation struct {
    ID             string    // local UUID
    TodoID         string    // Notion page UUID
    ConversationID string    // Jetbox cascade ID
    LaunchedAt     time.Time
    Mode           string    // "navigate" | "direct"
}

// Repository is the domain-level interface for conversation persistence.
// Handlers depend on this interface, not on the concrete pgx implementation.
type Repository interface {
    ListByTodo(ctx context.Context, todoID string) ([]Conversation, error)
    List(ctx context.Context) ([]Conversation, error)
    Create(ctx context.Context, c Conversation) (Conversation, error)
}

// NewRepository returns a Repository backed by q.
func NewRepository(q db.Querier) Repository {
    return &pgxRepository{q: q}
}

type pgxRepository struct{ q db.Querier }

func (r *pgxRepository) List(ctx context.Context) ([]Conversation, error) {
    rows, err := r.q.Query(ctx, `
        SELECT id, todo_id, conversation_id, launched_at, mode
        FROM todo_conversations
        ORDER BY launched_at DESC
    `)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    return scanConversations(rows)
}

func (r *pgxRepository) ListByTodo(ctx context.Context, todoID string) ([]Conversation, error) {
    rows, err := r.q.Query(ctx, `
        SELECT id, todo_id, conversation_id, launched_at, mode
        FROM todo_conversations
        WHERE todo_id = $1
        ORDER BY launched_at DESC
    `, todoID)
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    return scanConversations(rows)
}

func (r *pgxRepository) Create(ctx context.Context, c Conversation) (Conversation, error) {
    c.ID = newUUID()
    c.LaunchedAt = time.Now().UTC()
    _, err := r.q.Exec(ctx, `
        INSERT INTO todo_conversations (id, todo_id, conversation_id, launched_at, mode)
        VALUES ($1, $2, $3, $4, $5)
    `, c.ID, c.TodoID, c.ConversationID, c.LaunchedAt, c.Mode)
    if err != nil {
        return Conversation{}, err
    }
    return c, nil
}
```

The corresponding migration:

```sql
-- 0001_create_conversations.sql
CREATE TABLE IF NOT EXISTS todo_conversations (
    id              TEXT        PRIMARY KEY,
    todo_id         TEXT        NOT NULL,
    conversation_id TEXT        NOT NULL,
    launched_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    mode            TEXT        NOT NULL CHECK (mode IN ('navigate', 'direct'))
);

CREATE INDEX IF NOT EXISTS idx_todo_conversations_todo_id
    ON todo_conversations (todo_id);
```

### 4.4 HTTP handlers wired to the repository

```go
// Server holds the HTTP handlers for the TODO plugin.
type Server struct{ repo Repository }

func NewServer(repo Repository) *Server { return &Server{repo: repo} }

func (s *Server) HandleList(w http.ResponseWriter, r *http.Request) {
    convs, err := s.repo.List(r.Context())
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    writeJSON(w, http.StatusOK, convs)
}

func (s *Server) HandleCreate(w http.ResponseWriter, r *http.Request) {
    var req struct {
        TodoID         string `json:"todoId"`
        ConversationID string `json:"conversationId"`
        Mode           string `json:"mode"`
    }
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }
    created, err := s.repo.Create(r.Context(), Conversation{
        TodoID:         req.TodoID,
        ConversationID: req.ConversationID,
        Mode:           req.Mode,
    })
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }
    writeJSON(w, http.StatusCreated, created)
}
```

--------------------------------------------------------------------------------

## 5. Database topology (one DB per workspace vs. shared)

The "preventing database race conditions" research
([conversation dab8ea46](https://docs.google.com/...)) concluded that **each
workspace should have its own PostgreSQL database** (or schema).

Rationale:

-   Sidecars for different workspaces run as separate processes. A shared DB
    requires careful locking to avoid corruption on concurrent writes.
-   A workspace-scoped DB is simpler: no cross-workspace JOIN ever makes sense;
    dropping a workspace is a `DROP DATABASE`; connection-level isolation is
    free.
-   PostgreSQL schema-per-workspace (same server, different schema) is an
    acceptable alternative if spinning up a DB per workspace is expensive.

**Decision for v2:** one PostgreSQL **schema per workspace**, all sharing the
same server instance. The schema name is the workspace name (e.g., `head`,
`feat_x`). The DSN passed to `pgxpool.New` includes `search_path=<workspace>`.

This is enforced in `main.go`:

```go
dsn := fmt.Sprintf(
    "host=%s port=%d dbname=agent_manager user=%s password=%s search_path=%s sslmode=disable",
    pgHost, pgPort, pgUser, pgPassword, sanitizeSchema(workspaceName),
)
pool, err := pgxpool.New(ctx, dsn)
```

Migrations run against `search_path=<workspace>`, so each workspace gets its own
table set.

--------------------------------------------------------------------------------

## 6. Testability

| Level                 | Strategy                                            |
| --------------------- | --------------------------------------------------- |
| Unit (repository)     | Pass a `pgxmock` mock that satisfies `db.Querier`   |
| Integration (handler) | `httptest.NewRecorder` + repository fake            |
:                       : (`fakeRepository` struct satisfying the interface)  :
| End-to-end            | Real PostgreSQL via `testcontainers-go` or a shared |
:                       : local `postgres\:alpine`                            :

The `Querier` interface is the key seam. Because handlers receive a `Repository`
interface (not the concrete pgx implementation), fake repositories are trivial
to write:

```go
type fakeRepository struct{ convs []Conversation }
func (f *fakeRepository) List(_ context.Context) ([]Conversation, error) {
    return f.convs, nil
}
// ...
```

No database needed for handler unit tests.

--------------------------------------------------------------------------------

## 7. File layout

```
sidecar/
├── main.go                       # Pool init, migrate, wire plugins
├── BUILD                         # bazel build file
│
├── db/
│   ├── querier.go                # Querier interface + WithTx helper
│   ├── migrate/
│   │   ├── migrate.go            # Migration runner
│   │   ├── 0001_create_conversations.sql
│   │   └── ...
│   └── BUILD
│
└── todo/
    ├── repository.go             # Repository interface + pgxRepository
    ├── server.go                 # HTTP handlers (HandleList, HandleCreate)
    ├── types.go                  # Conversation domain struct
    ├── repository_test.go        # pgxmock-based unit tests
    ├── server_test.go            # httptest-based handler tests
    └── BUILD
```

Future plugins follow the same layout: `sidecar/<plugin>/`.

--------------------------------------------------------------------------------

## 8. Open questions

| Question                  | Recommendation                             |
| ------------------------- | ------------------------------------------ |
| Which PostgreSQL server?  | Local `postgres` process managed by the    |
:                           : gateway CLI, or a long-running system      :
:                           : postgres. The CLI should `pg_isready`      :
:                           : check it on startup and fail fast if       :
:                           : absent.                                    :
| Connection string source  | CLI flag `--pg-dsn` or per-workspace       |
:                           : config in                                  :
:                           : `~/.agent_manager/workspaces/<name>.json`. :
:                           : Flag is simpler for now.                   :
| pgx major version         | v5 (current). It changes the `Rows.Scan`   |
:                           : ergonomics vs. v4 — use `pgx/v5/pgxpool`.  :
| Transaction wrapping in   | Handlers that need atomicity call          |
: handlers                  : `db.WithTx(ctx, pool, func(q db.Querier)   :
:                           : error { ... })`. Repositories are          :
:                           : constructed inside the callback with       :
:                           : `todo.NewRepository(q)`.                   :
| `pgxmock` availability in | Check the module registry for `pgxmock` or |
: the project               : `pgx/v5/pgxmock`. If absent, use a         :
:                           : hand-written fake implementing             :
:                           : `db.Querier`.                              :

--------------------------------------------------------------------------------

## 9. Summary: what is actually being built

| Component                      | Necessary? | Rationale                     |
| ------------------------------ | ---------- | ----------------------------- |
| `db.Querier` interface         | ✅ Yes      | Testability seam; keeps pool  |
:                                :            : concrete, hidden from plugins :
| `db.WithTx` helper             | ✅ Yes      | Reusable transaction          |
:                                :            : boilerplate                   :
| `db/migrate` package           | ✅ Yes      | Schema evolution without an   |
:                                :            : external tool                 :
| `todo.Repository` interface    | ✅ Yes      | Decouples handler logic from  |
:                                :            : SQL; enables fakes in tests   :
| `todo.pgxRepository`           | ✅ Yes      | The one concrete              |
:                                :            : implementation doing the SQL  :
:                                :            : work                          :
| Generic "DB wrapper" struct    | ❌ No       | Pointless indirection;        |
:                                :            : pgxpool.Pool is already good  :
| Generic ORM / reflection-based | ❌ No       | Hand-written scan functions   |
: mapper                         :            : are simpler and type-safe     :
