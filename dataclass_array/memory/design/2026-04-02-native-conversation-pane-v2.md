# Design: Native Split-Pane Conversation View for TODO Attempts (v2)

Date: 2026-04-02 \
Status: **Proposed** \
Supersedes: `2026-04-02-native-conversation-pane.md`

--------------------------------------------------------------------------------

## Objective

When the user clicks an attempt in the TODO detail panel, open the conversation
directly in a resizable side pane **within the TODO view** instead of navigating
away. The result is a three-panel layout:

```
┌───────────────┬──────────────────────┬──────────────────────────────┐
│  Tree pane    │  Detail pane         │  Conversation pane           │
│  (unchanged)  │  todo meta + input   │  ConversationView (native)   │
└───────────────┴──────────────────────┴──────────────────────────────┘
```

--------------------------------------------------------------------------------

## Codebase Findings

### 1. Current attempt list behaviour

In `plugin.tsx` (`TodoDetail`, lines 971-985), each attempt is rendered as:

```tsx
<a href={`/${currentWs}/?cascadeId=${attempt.conversationId}`} ...>
  {attempt.conversationId}
</a>
```

This performs a **full-page navigation** away from the TODO view. The URL format
`?cascadeId=` is also wrong — the correct path is `/c/:cascadeId` as per the
router's `ROUTE_PATHS.conversation` definition.

### 2. React tree placement — plugin is already in context

The TODO plugin sidebar is rendered through `CustomViewRoute` inside
`ManagerContent`, which lives inside `ManagerContextProvider`. This means the
plugin already has access to:

-   Redux store (`useSelector`, `useDispatch`)
-   `ManagerSyncedStateContext` / `useTrajectorySummariesProvider`
-   `TrajectoriesContext` / `OptimisticTrajectoriesContext`
-   `AgentInputContext` (one shared instance per window)

No manual context bridging or iframe is needed.

### 3. `ConversationView` API

```tsx
// From ConversationView.tsx
export interface ConversationViewProps {
  cascadeId: string | undefined;
  disableFocusOnBgClick?: boolean;
}
```

It wraps everything in `CortexStepHandlerWrapper`, which: 1. Resolves the
workspace URI from the cascade ID via `useWorkspaceUriFromCascadeId(cascadeId)`.
2. Creates the step handler via `useStepHandlerImpl`. 3. Creates
`CortexStepHandlerProvider` for all child renderers.

**Conclusion:** `<ConversationView cascadeId={id} disableFocusOnBgClick />` is
portable and self-contained **as long as it is inside the existing React tree**.

### 4. ⚠️ The `AgentInputContext` singleton problem

`AgentInputProvider` creates a **single** `inputBoxRef` per context instance.
When `ConversationView` is embedded inside the TODO pane, it shares the same
`AgentInputContext` as the main conversation view. This causes two potential
issues:

**a. `NotificationCardPortal`** in `ConversationView` reads `inputBoxRef` and
tries to portal notifications above the input box — it will target the *wrong*
input box (the one in the currently-active main conversation, not the embedded
view).

**b. `useInputFocusOnBgClick`** is already guarded by `disableFocusOnBgClick`,
so passing that prop eliminates issue (b). ✅

**Mitigation for (a):** The `NotificationCardPortal` renders `null` when
`portalTarget` is `null`, which happens when `inputBoxRef.current` is `null`. In
the embedded view there is **no `AgentManagerInputBox`** mounted — the input box
renders only for the active conversation in the main area. So
`inputBoxRef.current` will be `null` from the embedded view's perspective, and
the portal becomes a no-op. ✅

The `CortexStepHandlerWrapper` inside the embedded pane reads the same
`inputBoxRef` as the main conversation. The `lexicalEditor` captured inside
`renderersWithCascadeId` will refer to the main input box. This is acceptable
for a read-only view — the markdown renderer's "apply to editor" functionality
will paste into whichever editor is currently focused, which is the correct
behaviour.

**Alternative (rejected):** Wrap `ConversationPane` in its own
`AgentInputProvider`. This would give a clean `inputBoxRef` but break the
lexical draft-sharing mechanism. Since the embedded pane is read-only, this is
unnecessary complexity.

### 5. `useWorkspaceUri` vs `useWorkspaceUriFromCascadeId`

`ConversationView` internally calls `useWorkspaceUri()`, which reads
`useSectionId()` → URL `?section=` param. In the TODO custom view, the route is
`/view/todos`, so `sectionId` is `undefined` and `useWorkspaceUri()` returns
`undefined`.

`CortexStepHandlerWrapper` handles this gracefully: it calls
`useWorkspaceUriFromCascadeId(cascadeId)` first, which looks up the workspace
from the Redux conversation store. For existing conversations this resolves
correctly. If the workspace is still `undefined` (conversation not yet in
Redux), it falls back to `fallbackWorkspaceUri` (which will also be `undefined`
here — acceptable since the trajectory will still load once SSE delivers the
data). ✅

### 6. `useSubagentParam` inside embedded view

`ConversationView` calls `useSubagentParam()`, which uses `useMatchRoute()` from
TanStack Router. On the `/view/todos` route this returns `undefined`, so
`effectiveCascadeId = cascadeId` (the passed prop). Correct behaviour. ✅

### 7. Routing — no URL changes needed for MVP

The embedded pane is driven by **local React state** (`selectedConversationId`
in `TodosPane`). The TODO route stays at `/view/todos`. No URL changes are
required.

Optional enhancement (post-MVP): persist the selected conversation ID as a URL
search param (`/view/todos?conversationId=...`) so that refreshing restores the
panel.

### 8. `ResizableSplitView` — nesting pattern

The existing `TodosPane` already uses one `ResizableSplitView`:

```
[Tree | Detail]   (axis=Row, baseSizing={280, 'first'})
```

The proposed layout nests a second `ResizableSplitView` inside the `second`
slot:

```
[Tree | [Detail | Conversation]]
```

This is valid — `ResizableSplitView` is a generic layout primitive. The inner
split uses `disable={!selectedConversationId}` so the sash disappears when no
conversation is open.

--------------------------------------------------------------------------------

## Proposed Implementation

### Files to change

| File                       | Change                                   |
| -------------------------- | ---------------------------------------- |
| `plugins/todos/plugin.tsx` | New `ConversationPane` component; layout |
:                            : change; state; import                    :

No sidecar changes. No BUILD changes. No new files required.

--------------------------------------------------------------------------------

### 1. New import

Add to `plugin.tsx` imports:

```tsx
import {ConversationView} from '@/AntigravityAgent/views/conversation/ConversationView';
```

### 2. New `ConversationPane` component

Add after `TodoDetail` (around line 999):

```tsx
// ── ConversationPane ────────────────────────────────────────────────────────────

interface ConversationPaneProps {
  conversationId: string;
  onClose: () => void;
}

/**
 * A thin wrapper around ConversationView that adds a header bar with a
 * close button. Rendered as the third pane in the TODO layout.
 *
 * ConversationView is self-contained: it creates its own
 * CortexStepHandlerWrapper and resolves the workspace URI from the
 * cascade ID via Redux. No additional context wiring is needed.
 */
function ConversationPane({conversationId, onClose}: ConversationPaneProps) {
  return (
    <div className="flex flex-col h-full border-l border-gray-500/20 bg-background">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-gray-500/20 shrink-0">
        <span
          className="text-xs font-mono text-description-foreground truncate opacity-70"
          title={conversationId}>
          {conversationId}
        </span>
        <button
          onClick={onClose}
          title="Close conversation"
          className="p-0.5 rounded hover:bg-list-hover transition-colors shrink-0">
          <GoogleSymbol name="close" size={14} className="opacity-70" />
        </button>
      </div>
      {/* Native conversation view */}
      <div className="flex-1 min-h-0">
        <ConversationView cascadeId={conversationId} disableFocusOnBgClick />
      </div>
    </div>
  );
}
```

### 3. State changes in `TodosPane`

Add one new state variable inside `TodosPane`:

```tsx
const [selectedConversationId, setSelectedConversationId] = useState<
  string | undefined
>();
```

Clear the conversation pane when the selected TODO changes:

```tsx
// Clear conversation pane when TODO selection changes
useEffect(() => {
  setSelectedConversationId(undefined);
}, [selectedId]);
```

### 4. Layout change in `TodosPane`

Replace the `second` slot of the outer `ResizableSplitView` with a nested split
view:

```tsx
second={
  <ResizableSplitView
    axis={SplitViewAxis.Row}
    baseSizing={{size: 480, for: 'second'}}
    minSize={300}
    minFillSize={280}
    disable={!selectedConversationId}
    debugLabel="todos-convo-split"
    first={
      <TodoDetail
        todo={selectedTodo}
        onTodoUpdate={handleTodoUpdate}
        autoFocusTitle={selectedTodo?.id === newlyCreatedId}
        onAttemptCreated={(attempt) => {
          // Also auto-open the new conversation on creation
          setSelectedConversationId(attempt.conversationId);
        }}
        onAttemptSelected={setSelectedConversationId}
      />
    }
    second={
      selectedConversationId ? (
        <ConversationPane
          conversationId={selectedConversationId}
          onClose={() => setSelectedConversationId(undefined)}
        />
      ) : (
        <div />
      )
    }
  />
}
```

### 5. `TodoDetail` — replace `<a>` with a button + add `onAttemptSelected` prop

Extend `TodoDetailProps`:

```tsx
interface TodoDetailProps {
  todo: Todo | undefined;
  onTodoUpdate: (updated: Todo) => void;
  autoFocusTitle?: boolean;
  onAttemptCreated?: (attempt: TodoAttempt) => void;
  onAttemptSelected?: (conversationId: string) => void;  // ← NEW
}
```

Replace the attempt `<a>` tag with a `<button>`:

```tsx
// BEFORE
<a
  href={`/${currentWs}/?cascadeId=${attempt.conversationId}`}
  className="font-mono text-xs text-blue-400 hover:underline truncate flex-1"
  title={attempt.conversationId}>
  {attempt.conversationId}
</a>

// AFTER
<button
  onClick={() => onAttemptSelected?.(attempt.conversationId)}
  className="font-mono text-xs text-blue-400 hover:underline truncate flex-1 text-left"
  title={attempt.conversationId}>
  {attempt.conversationId}
</button>
```

Optionally, highlight the currently-open attempt:

```tsx
// Add selectedConversationId prop or receive it from parent
// Row wrapper class change:
className={`flex items-center ... ${
  attempt.conversationId === selectedConversationId
    ? 'ring-1 ring-blue-400/30'
    : 'bg-editor-background hover:bg-list-hover'
}`}
```

> **Note:** Since `selectedConversationId` lives in `TodosPane` and `TodoDetail`
> is a child, there are two options: 1. Pass `selectedConversationId` down as a
> prop to `TodoDetail` (simplest). 2. Keep `TodoDetail` stateless with respect
> to the selection and handle highlighting in `TodosPane`. Option 1 is
> recommended.

--------------------------------------------------------------------------------

## Sizing strategy for the inner split

```
baseSizing={{size: 480, for: 'second'}}
```

-   The conversation pane starts at 480 px wide (roughly half a 960 px detail
    area).
-   `minSize={300}`: conversation pane cannot shrink below 300 px.
-   `minFillSize={280}`: detail pane cannot shrink below 280 px.
-   `disable={!selectedConversationId}`: when no conversation is open, the inner
    split is disabled and the sash is hidden.

Double-clicking the sash resets to the 480 px base (built into
`ResizableSplitView`). ✅

--------------------------------------------------------------------------------

## Edge cases and mitigations

| Case                          | Behaviour                              |
| ----------------------------- | -------------------------------------- |
| User selects a different TODO | `selectedConversationId` clears via    |
:                               : `useEffect`; pane collapses            :
| User closes pane via ✕        | `setSelectedConversationId(undefined)` |
:                               : collapses pane                         :
| Conversation not yet loaded   | `ConversationView` shows               |
:                               : `ConversationLoadingView` spinner      :
| Cascade ID not in Redux       | `useWorkspaceUriFromCascadeId` returns |
:                               : `undefined`; step handler initialises  :
:                               : without workspace URI; trajectory      :
:                               : loads via SSE when available           :
| Two conversations in sequence | Clicking a different attempt updates   |
:                               : `selectedConversationId`;              :
:                               : `ConversationView` re-renders with new :
:                               : cascade ID                             :
| `inputBoxRef` conflicts       | `disableFocusOnBgClick` prevents focus |
:                               : theft; `NotificationCardPortal` is a   :
:                               : no-op (no input box mounted in         :
:                               : embedded view)                         :
| `ResizableSplitView` with     | Sash hidden; `<div />` fills `second`  |
: `disable=true`                : slot with zero effective width         :
| Window too narrow             | `minFillSize={280}` on the outer split |
:                               : prevents tree pane from being squeezed :
:                               : below 280 px                           :
| New attempt created via       | Auto-opens the conversation pane via   |
: `TodoChatInput`               : `onAttemptSelected` callback           :

--------------------------------------------------------------------------------

## What is NOT done here (future work)

-   **URL persistence**: encoding `?conversationId=` in the URL so refreshing
    restores the pane (requires extending the TanStack Router search params
    schema).
-   **Real-time status badges on attempt rows**: showing a live "Running…"
    indicator (requires hooking into `TrajectorySummariesProvider` — prototyped
    in conversation `7ceca9c3-ccb4-4fa9-9d02-479ea32b19e6`).
-   **A second input box inside `ConversationPane`**: sending new messages from
    within the embedded pane. `ConversationView` already renders
    `ActiveInputArea` internally, so this may work out of the box once the
    workspace URI resolves. Needs verification.
