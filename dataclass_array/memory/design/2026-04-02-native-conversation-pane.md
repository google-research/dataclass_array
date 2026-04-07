# Research: Native Split-Pane Conversation View for TODO Attempts

Date: 2026-04-02

## Objective

Investigate how to open the `ConversationView` natively inside the TODO plugin's detail view (in a split pane) instead of navigating away or using an iframe.

---

## Findings

### 1. Context Availability in Plugins

The TODO plugin sidebar is rendered via `CustomViewRoute` inside the main `ManagerContent` (and thus inside `ManagerContextProvider`).

This means the plugin **already resides within the same React tree** as the rest of the application. It has access to:
- Redux store (state, dispatch)
- `ManagerSyncedStateContext`
- `TrajectoriesContext` and `TrajectoriesOptimisticContext`

We do **not** need to manually propagate these contexts or use `lsClient` for state synchronization; standard React context resolution works out of the box.

### 2. `ConversationView` and `CortexStepHandlerWrapper`

The `ConversationView` component (from `@/AntigravityAgent/views/conversation/ConversationView`) handles its own step handler context setup via internal usage of `CortexStepHandlerWrapper`.

```tsx
// Inside ConversationView.tsx
<CortexStepHandlerWrapper cascadeId={effectiveCascadeId} ...>
  ...
  <TrajectoryOrLoadingSpinner cascadeId={effectiveCascadeId} />
  ...
</CortexStepHandlerWrapper>
```

This makes it highly portable: we can drop `<ConversationView cascadeId={...} />` into any layout that is already inside the `ManagerContextProvider` (which the Todos plugin is).

### 3. URL Schema Correction

The URL for conversations is `/c/:cascadeId` (e.g., `/head/c/562a970d-5f45-4cbd-b168-416c1685a398`), not `?cascadeId=`.

Since we are opening the conversation *within* the TODO view (retaining the `/view/todos` route), we do not need to push a new URL to the main router. We can manage the active conversation ID purely via local component state or URL search parameters (e.g., `/view/todos?conversationId=...`).

---

## Proposed Design

We will modify `plugin.tsx` in the `todos` plugin to use a nested `ResizableSplitView`.

### Layout

```
[ Tree Pane | Detail Pane ]
```
becomes
```
[ Tree Pane | [ Detail Pane | Conversation Pane ] ]
```

### State

Add `selectedConversationId` state to `TodosPane`.

### Component Composition

```tsx
<ResizableSplitView axis={SplitViewAxis.Row} ...>
  first={<TodoTree ... />}
  second={
    <ResizableSplitView axis={SplitViewAxis.Row} ... disable={!selectedConversationId}>
      first={<TodoDetail ... />}
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
/>
```

The `ConversationPane` will be a simple wrapper that renders a header bar (with close button) and the native `ConversationView`.

```tsx
function ConversationPane({conversationId, onClose}: ConversationPaneProps) {
  return (
    <div className="flex flex-col h-full border-l border-gray-500/20">
      <div className="flex items-center justify-between p-2 border-b border-gray-500/20">
        <span className="text-xs font-mono truncate">{conversationId}</span>
        <button onClick={onClose}><GoogleSymbol name="close" /></button>
      </div>
      <div className="flex-1 overflow-hidden">
        <ConversationView cascadeId={conversationId} disableFocusOnBgClick />
      </div>
    </div>
  );
}
```
