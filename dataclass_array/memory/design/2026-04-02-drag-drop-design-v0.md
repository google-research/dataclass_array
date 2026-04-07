# Drag-and-Drop Todo Nesting

Add drag-and-drop reordering and nesting to the `TodoTree`. Users can: - **Drop
onto an item** → make the dragged item a child of the target. - **Drop between
items** → insert the dragged item at that position (same parent as the gap).

No new backend data model is needed — the existing `parent_id` on `todos`
already encodes nesting. We need a `position` field for ordering, plus two new
PATCH semantics.

--------------------------------------------------------------------------------

## User Review Required

> [!IMPORTANT] The schema currently has no `position` column — todos have no
> explicit ordering. We need to decide whether to add an explicit `position` INT
> (fractional/lexicographic ordering like Notion) or keep ordering implicit
> (e.g., `created_at`). The plan below uses an explicit `position` float, which
> is the simplest approach that doesn't require re-numbering siblings.

> [!NOTE] No drag-and-drop library is available in the GoB repo today. We will
> use the **native HTML5 Drag and Drop API** (`draggable`, `ondragstart`,
> `ondragover`, `ondrop`) — zero deps, works in all modern browsers, and matches
> the existing no-extra-deps pattern.

--------------------------------------------------------------------------------

## Proposed Changes

### Backend — Schema & Repo

#### [MODIFY] [schema.sql](file:///drag-drop-todo-nesting/agent_manager/v2/plugins/todos/sidecar/schema.sql)

Add `position FLOAT8 NOT NULL DEFAULT 0` to `todos`. New todos get `position =
MAX(sibling) + 1`.

```sql
ALTER TABLE todos ADD COLUMN IF NOT EXISTS position FLOAT8 NOT NULL DEFAULT 0;
```

New `PATCH /api/todos/{id}/move` endpoint accepts: `json { "parentId": "<id or
null>", "position": 1.5 }` Returns the updated `Todo`.

#### [MODIFY] [types.go](file:///drag-drop-todo-nesting/agent_manager/v2/plugins/todos/sidecar/types.go)

Add `Position float64` field to `Todo`.

#### [MODIFY] [repo.go](file:///drag-drop-todo-nesting/agent_manager/v2/plugins/todos/sidecar/repo.go)

-   `CreateTodo`: compute `position = MAX(sibling position) + 1` (or `1` if no
    siblings).
-   Add `MoveTodo(ctx, id, parentID *string, position float64)` function.
-   `ListTodos`: `ORDER BY position ASC` (within each level).

#### [MODIFY] [plugin.go](file:///drag-drop-todo-nesting/agent_manager/v2/plugins/todos/sidecar/plugin.go)

Register `PATCH /api/todos/{id}/move` handler calling `MoveTodo`. Broadcast a
`todo_updated` SSE event after the move.

--------------------------------------------------------------------------------

### Frontend — Drag-and-Drop UI

#### [MODIFY] [plugin.tsx](file:///drag-drop-todo-nesting/agent_manager/v2/plugins/todos/plugin.tsx)

Changes are confined to `TodoTreeNode` and a new `DropZone` inline component.
`TodoTree` and `TodosPane` are untouched.

**Key design decisions (api-design aligned):**

| Concept                | Decision                                            |
| ---------------------- | --------------------------------------------------- |
| Drop target = item row | Visual highlight on `dragover`; drop → `parentId =  |
:                        : target.id`                                          :
| Drop target = gap      | Thin separator bar between rows; drop → `parentId = |
:                        : target.parentId`, `position` between neighbors      :
| State                  | Single `draggingId: string \| undefined` in         |
:                        : `TodoTree` (no global)                              :
| API call               | `PATCH /{ws}/api/todos/{id}/move` — clear semantic, |
:                        : separate from content PATCH                         :
| Optimistic update      | Immediately reorder local state; revert on error    |
| SSE                    | `todo_updated` event already handled — cross-tab    |
:                        : moves propagate for free                            :

**Component structure:**

```
TodoTree
  └── TodoTreeNode (draggable)
        ├── DropZone (gap above — "insert before")
        ├── <row> (drop target → "make child")
        └── children (recursive)
              └── DropZone (gap below last child — "insert after")
  └── DropZone (gap at bottom — "insert as root at end")
```

**Visual feedback:** - Dragging item: `opacity-40` on the source row. -
`dragover` on item row: `ring-2 ring-blue-400 rounded` highlight. - `dragover`
on gap: gap expands to `h-1.5 bg-blue-400/40 rounded`.

**`useMoveHook`** (new, separate from `useSaveStatus`): `ts function
useMoveTodo(ws: string): { move: (id: string, parentId: string | undefined,
position: number) => Promise<Todo | null> }` One clear responsibility: call the
move API. `TodoTree` calls this and updates local state.

**`buildTree` update:** sort children by `position` (ascending) so tree order
matches DB order.

--------------------------------------------------------------------------------

## Open Questions

> [!IMPORTANT] **Fractional positioning strategy**: When dropping between item A
> (position=1) and B (position=2), we set `position=(1+2)/2 = 1.5`. This can
> degrade after many moves. Should we normalize positions after every move, or
> use a larger base (e.g. step 1000 and normalize when gap < 0.001)? The plan
> defaults to "normalize lazily" (only when gap < 1e-6).

> [!NOTE] **Cross-level drop**: If the user drops an ancestor onto one of its
> own descendants, the move is invalid. We guard against this on the frontend
> (don't show drop zones inside the dragged subtree) AND on the backend (cycle
> detection before committing).

--------------------------------------------------------------------------------

## Verification Plan

### Automated Tests

-   Frontend unit tests: `TodoTreeNode` with drag events mocked via
    `fireEvent.dragStart`, `fireEvent.dragOver`, `fireEvent.drop`.
-   Backend: `repo_test.go` — `TestMoveTodo` covering: nest, insert-before,
    insert-after, cycle guard.

```bash
./agent_manager.py test --test-frontend --test-sidecar
```

### Manual Verification (via gbrowser)

1.  Drag item A onto item B → A appears indented under B.
2.  Drag between two root items → item moves between them at root level.
3.  Drag item to its own descendant → rejected (no visual drop zone shown).
4.  Open two tabs — move in tab 1 → SSE pushes `todo_updated` → tab 2 reorders
    immediately.
