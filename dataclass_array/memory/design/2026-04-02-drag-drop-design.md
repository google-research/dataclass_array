# Drag-and-Drop Todo Nesting — Design

> Date: 2026-04-02

## Goal

Add drag-and-drop to `TodoTree`: - **Drop onto a row** → dragged item becomes a
child of the target. - **Drop into a gap** → dragged item is inserted between
two siblings (same parent).

--------------------------------------------------------------------------------

## 1. Ordering: Lexicographic String Keys

### Why not floats?

Float positions need normalization when the gap shrinks below floating-point
precision (e.g. after 50 bisections between 1.0 and 2.0). That means periodic
background re-numbering or a lazy GC pass.

### Lexicographic keys

Use variable-length lowercase-letter strings ordered lexicographically:

```
"a"  <  "b"  <  "c"  ...
"a"  <  "aa" <  "ab" ...
"n"  is the midpoint of "" and ""   (initial root)
```

**Midpoint algorithm** between string `lo` and string `hi` (both in alphabet
`a–z`):

1.  Pad the shorter string with the lowest character (`a`) until lengths match.
2.  Treat each character as a base-26 digit.
3.  Compute the integer midpoint digit-by-digit, carrying if needed.
4.  If `lo + 1 == hi` (no room), append a middle character to `lo` (e.g.
    `"ab"` + `"n"` → `"abn"`).

**Key space is infinite** in practice: strings grow one character at a time only
when the interval is exhausted at that length. For a personal todo list, this
never happens.

**Alphabet choice:** lowercase `a–z` (26 chars). The midpoint character is `n`
(index 13 of 0–25).

**Initial positions:** - First root todo: `"n"` - Second root todo: `"t"`
(midpoint of `"n"` and `"z"`... actually just `MAX(siblings) + "n"`) - New todo
at end of sibling group: `MAX(sibling_position) + "n"` (string concatenation,
which is lexicographically larger).

### Schema change

```sql
ALTER TABLE todos ADD COLUMN IF NOT EXISTS position TEXT NOT NULL DEFAULT 'n';
```

`ORDER BY position ASC` within each sibling group (same `parent_id`).

--------------------------------------------------------------------------------

## 2. Does This Change SSE Events?

**No — the SSE event shape is unchanged.**

`todo_updated` already carries the full `Todo` object:

```go
// events.go — unchanged
TodoEvent{Type: EventTodoUpdated, Todo: &updated}
```

The `Todo` struct gains a `Position string` field, which is included in the JSON
payload. The frontend already replaces the full todo on `todo_updated`:

```ts
case 'todo_updated':
  setTodos(prev => prev.map(t => t.id === event.todo.id ? event.todo : t));
```

The only change: after applying a `todo_updated` event the local list must be
**re-sorted by position**. Today `buildTree` doesn't sort (it relies on DB
ordering). We add one sort step in `buildTree` or in `handleTodoEvent`.

**Summary of SSE impact:** | What | Changed? | Why | |---|---|---| | Event types
(`todo_created`, `todo_updated`, `todo_deleted`) | ✗ | | | Event wire format
(`data: <json>\n\n`) | ✗ | | | `Todo` JSON payload | ✓ | adds `position: string`
field | | Frontend event handler shape | ✗ | | | Re-sort after `todo_updated` |
✓ | needed to reflect new position |

--------------------------------------------------------------------------------

## 3. API Design

### Core principle (from `api-design` skill)

> Code should match high-level semantics. Never mix levels of abstraction.

The client thinks in terms of **drag intent**: "I dropped A after B, under
parent P". It must **not** think in terms of lexicographic key computation —
that's a storage detail.

→ The server owns the key generation. The client sends semantic intent.

### New endpoint: `PATCH /api/todos/{id}/move`

```
PATCH /api/todos/{id}/move
Content-Type: application/json

{
  "parentId": "<uuid> | null",   // null = move to root
  "afterId":  "<uuid> | null"    // null = insert at the beginning of siblings
}
```

**Semantics:** - `parentId` — new parent (or `null` for root). Required. -
`afterId` — the sibling after which to insert. `null` = insert before all
current siblings.

The server: 1. Validates `parentId` exists (or is null). 2. Guards against
cycles: `id` must not be an ancestor of `parentId`. 3. Fetches the `position` of
`afterId` (call it `lo`) and the next sibling (call it `hi`). 4. Computes
`midpoint(lo, hi)` using the lexicographic algorithm. 5. Updates `todos SET
parent_id = $parentId, position = $computed WHERE id = $id`. 6. Broadcasts
`todo_updated` SSE event. 7. Returns the updated `Todo` (200 OK).

**Why `afterId` instead of `position`?**

| Option                  | Who computes the key? | Who owns the invariant? |
| ----------------------- | --------------------- | ----------------------- |
| Client sends `position: | Client                | Client (fragile)        |
: "abn"`                  :                       :                         :
| Client sends `afterId:  | **Server**            | **Server** (correct)    |
: "<uuid>"`               :                       :                         :

Sending `afterId` is the high-level semantic ("insert after this item"). The
server translates it to a key — the client is decoupled from the ordering
algorithm entirely.

### Routing

`handleTodoByID` currently routes `PATCH /api/todos/{id}` to content edits. Move
is a **different operation** (different fields, different invariants), so it
deserves its own route:

```
/api/todos/{id}        → content PATCH (title, content, status)
/api/todos/{id}/move   → structural PATCH (parentId, afterId)
```

This keeps each handler single-purpose (api-design: "one feature per file / one
clear action").

In `plugin.go`, `handleTodoByID` currently rejects paths containing `/`. We
update it to forward `/api/todos/{id}/move` to a dedicated handler **before**
the slash check:

```go
func (p *Plugin) handleTodoByID(w http.ResponseWriter, r *http.Request) {
    path := strings.TrimPrefix(r.URL.Path, "/api/todos/")

    if rest, ok := strings.CutSuffix(path, "/move"); ok {
        p.handleMoveTodo(w, r, rest)   // id = rest
        return
    }

    id := path
    if id == "" || strings.Contains(id, "/") { ... }
    ...
}
```

### Abstraction layers

```
plugin.go (HTTP handler)
  └─ handleMoveTodo(w, r, id)
        Validates input, calls repo, broadcasts SSE
        └─ repo.moveTodo(ctx, id, parentID *string, afterID *string) (Todo, error)
              Computes new position via lexkey.Midpoint
              Issues UPDATE in a single transaction
              └─ lexkey.Midpoint(lo, hi string) string   [new pure package]
```

`lexkey` is a tiny standalone package — pure function, no DB dependency,
trivially testable.

--------------------------------------------------------------------------------

## 4. Frontend Abstractions

```
TodosPane
  └─ state: todos [], draggingId ?string
  └─ useMoveTodo(ws)            ← new hook, one responsibility: call the move API
  └─ TodoTree
       └─ TodoTreeNode (draggable attribute)
            ├─ <DropZone position="before" />   ← renders gap + handles dragover/drop
            ├─ <row onDragOver onDrop />        ← drop-onto → make child
            └─ children
                  └─ <DropZone position="after" /> ← gap after last child
```

**`useMoveTodo`:**

```ts
function useMoveTodo(ws: string): {
  move: (id: string, parentId: string | undefined, afterId: string | undefined)
      => Promise<Todo | null>;
}
```

**`DropZone`** (gap component):

```ts
interface DropZoneProps {
  parentId: string | undefined;  // what parent the inserted item will have
  afterId:  string | undefined;  // what sibling it goes after (undefined = insert first)
  draggingId: string | undefined;
  onDrop: (parentId: string | undefined, afterId: string | undefined) => void;
}
```

The gap knows its own semantic position. When dropped on, it calls `onDrop` with
the right `parentId`/`afterId` pair. `TodoTreeNode` doesn't compute positions —
it just wires props.

**Optimistic update**: on drop, immediately reorder `todos` in local state (set
`dragged.parentId` and re-sort by estimated position). If the server returns an
error, revert to the pre-drag snapshot.

--------------------------------------------------------------------------------

## 5. Cycle Guard

Backend (authoritative):

```sql
-- PostgreSQL recursive CTE to check if 'candidate_parent' is a descendant of 'id'
WITH RECURSIVE descendants AS (
  SELECT id FROM todos WHERE id = $id
  UNION ALL
  SELECT t.id FROM todos t
  JOIN descendants d ON t.parent_id = d.id
)
SELECT 1 FROM descendants WHERE id = $candidateParentId
```

If any row returned → reject with `400 Bad Request: "cannot nest a todo under
its own descendant"`.

Frontend (UX-only guard, does not replace backend): - While dragging item X,
mark all descendants of X with `data-no-drop`. Drop zones inside those nodes
render as invisible and ignore `dragover`/`drop` events.

--------------------------------------------------------------------------------

## 6. Summary of Changes

| Layer    | File(s)                    | Change                              |
| -------- | -------------------------- | ----------------------------------- |
| Schema   | `schema.sql`               | Add `position TEXT NOT NULL DEFAULT |
:          :                            : 'n'`                                :
| Types    | `types.go`                 | Add `Position string` to `Todo`     |
| Ordering | `lexkey/lexkey.go` *(new)* | `Midpoint(lo, hi string) string`    |
| Repo     | `repo.go`                  | `moveTodo`, `createTodo` (compute   |
:          :                            : initial position), `listAllTodos`   :
:          :                            : (ORDER BY position)                 :
| HTTP     | `plugin.go`                | `handleMoveTodo` + route            |
:          :                            : `/api/todos/{id}/move`              :
| Frontend | `plugin.tsx`               | `useMoveTodo`, `DropZone`, drag     |
:          :                            : state, `buildTree` sort by position :
