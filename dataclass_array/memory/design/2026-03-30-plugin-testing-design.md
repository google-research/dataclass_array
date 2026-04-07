# Plugin Testing Design

**Date:** 2026-03-30 **Author:** epot

--------------------------------------------------------------------------------

## 1. Problem statement

The plugin system lives in **XX** (`plugins/` + `plugins-core/`) but the
test infrastructure lives in the **YY** (`exa/agent_ui_toolkit/`). Our
plugins are compiled *into* the GoB bundle via Vite aliases (`@plugins →
.../plugins/`, `@plugins-core → .../plugins-core/`), so at build time they are
just ordinary TypeScript modules inside the GoB project.

This split creates three testing risks identified during the grey-screen
incident:

| Risk                  | Trigger mechanism        | What failed            |
| --------------------- | ------------------------ | ---------------------- |
| Missing Redux context | Plugin `effect` uses     | `ConversationRecorder` |
:                       : `useSelector`            : crashed in isolation   :
| Wrong injection point | `<PluginEffects>` placed | Structural regression, |
:                       : outside provider         : grey screen            :
| Silent no-op wiring   | `wirePlugins` called     | Sidebar contributions  |
:                       : before featureManager    : swallowed              :
:                       : init                     :                        :

TypeScript doesn't catch any of these — they are *runtime invariant* violations.
We need actual component tests.

--------------------------------------------------------------------------------

## 2. Available test infrastructure (GoB repo)

The GoB repo already has a fully configured, working Vitest + React Testing
Library setup with **jsdom** and no network access:

```
exa/agent_ui_toolkit/
├── vitest.config.ts          # jsdom env, globals:true, @testing-library/react
├── vitest.setup.ts           # jest-dom matchers, ResizeObserver mock, matchMedia mock
├── node_modules/
│   ├── vitest@4              # ✅ test runner
│   ├── @testing-library/react@16  # ✅ render / screen / fireEvent
│   ├── @testing-library/jest-dom  # ✅ custom matchers
│   ├── @reduxjs/toolkit       # ✅ configureStore
│   └── react-redux            # ✅ Provider
```

**`npm run test`** (`vitest run`) already picks up every file matching
`src/**/*.test.{ts,tsx}` | `dev/**/*.test.{ts,tsx}`.

The Vite *build* and the Vitest *test* runner both resolve `@plugins` and
`@plugins-core` via aliases, but **the vitest config currently has neither
alias**. This is the only gap to bridge.

--------------------------------------------------------------------------------

## 3. The alias gap — and how to close it

### 3a. Why the gap exists

`vitest.config.ts` only configures:

```ts
include: ['src/**/*.test.{ts,tsx}', 'dev/**/*.test.{ts,tsx}', ...]
// No resolve.alias for @plugins or @plugins-core
```

Our plugin tests import from `@plugins/...` or `../../plugins/...` (relative).
The relative path can work if tests sit inside the worktree where plugins are
symlinked (see §4), but `@plugins` aliases fail without explicit configuration.

### 3b. Proposed approach — patch `vitest.config.ts`

We already have a patch system (`patches/`). The cleanest approach is a new
patch file: **`patches/plugins_vitest.patch`**.

The patch adds `resolve.alias` entries pointing to the same absolute source
paths that `vite.config.ts` uses:

```diff
--- exa/agent_ui_toolkit/vitest.config.ts
+++ exa/agent_ui_toolkit/vitest.config.ts
+import path from 'path';
+
 export default defineConfig({
   test: {
     globals: true,
     environment: 'jsdom',
     setupFiles: ['./vitest.setup.ts'],
     include: [
       'src/**/*.test.{ts,tsx}',
       'dev/**/*.test.{ts,tsx}',
+      '../../..//agent_manager/v2/plugins/**/*.test.{ts,tsx}',
+      '../../..//agent_manager/v2/plugins-core/**/*.test.{ts,tsx}',
     ],
   },
+  resolve: {
+    alias: {
+      '@plugins': '__PLUGINS_DIR__',
+      '@plugins-core': '__PLUGINS_CORE_DIR__',
+      '@': path.resolve(__dirname, 'src'),
+    },
+  },
 });
```

`__PLUGINS_DIR__` and `__PLUGINS_CORE_DIR__` are already template placeholders
substituted by `builder.py` when applying patches (same as in
`plugins_vite.patch.tpl`).

This means: - Tests run from **inside the worktree** (where `npm run test`
already works). - Plugin test files live in **the source tree** and are included via the
`include` glob. - No new toolchain, no new npm packages.

### 3c. Alternative — keep tests closer to source

Instead of reaching into the source tree from the vitest include path, test files can
live in the **worktree itself** (created by the build step). The builder would
copy / symlink a `tests/` directory from `v2/plugins/` into the worktree. This
is more flexible but adds builder complexity. Recommended only if the absolute
path glob proves fragile.

--------------------------------------------------------------------------------

## 4. Test file structure and location

### Convention

Test files live **next to the source files they test** in the source tree:

```
v2/
├── plugins-core/
│   ├── PluginEffects.tsx
│   ├── PluginEffects.test.tsx     ← NEW
│   ├── wire.ts
│   └── wire.test.ts               ← NEW
└── plugins/
    └── conversations/
        ├── plugin.tsx
        └── plugin.test.tsx        ← NEW
```

This is idiomatic for Vitest (similar to Jest's `*.test.tsx` convention used
throughout the GoB repo) and keeps tests co-located with what they verify.

### Naming

-   `<module>.test.tsx` for React component tests.
-   `<module>.test.ts` for pure logic tests.

--------------------------------------------------------------------------------

## 5. Test categories and exact test cases

### 5a. Component isolation tests (`plugins-core/PluginEffects.test.tsx`)

**Goal:** Verify `PluginEffects` can be rendered in isolation (no context
required) and that it mounts effects from all plugins.

```tsx
import {render} from '@testing-library/react';
import {describe, it, expect, vi} from 'vitest';
import {PluginEffects} from './PluginEffects';
import type {Plugin} from './types';

describe('PluginEffects', () => {
  it('renders without crashing with an empty plugin list', () => {
    // PluginEffects itself needs no context — it just renders children.
    render(<PluginEffects plugins={[]} />);
  });

  it('mounts an effect component for every plugin that declares one', () => {
    const mounted: string[] = [];
    const EffectA = () => { mounted.push('a'); return null; };
    const EffectB = () => { mounted.push('b'); return null; };
    const plugins: Plugin[] = [
      {id: 'a', effect: EffectA},
      {id: 'b', effect: EffectB},
      {id: 'c'},  // No effect — should be skipped.
    ];
    render(<PluginEffects plugins={plugins} />);
    expect(mounted).toEqual(['a', 'b']);
  });
});
```

**Why this matters:** If `PluginEffects` accidentally acquired a Redux
dependency in the future, this test would fail in isolation and surface the
regression immediately.

### 5b. Plugin context requirement tests (`plugins/conversations/plugin.test.tsx`)

**Goal:** Document and enforce that `ConversationRecorder` *requires* a Redux
`Provider`. Written as a "crash test" ─ the component must throw (or
console.error) without a store, and must not throw with one.

```tsx
import {render} from '@testing-library/react';
import {describe, it, expect, vi, beforeEach, afterEach} from 'vitest';
import {Provider} from 'react-redux';
import {configureStore} from '@reduxjs/toolkit';
import {conversationSlice} from '@/common/...';  // or a minimal stub
import {conversationsPlugin} from './plugin';

describe('ConversationRecorder (conversations plugin effect)', () => {
  let consoleError: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
  });
  afterEach(() => { consoleError.mockRestore(); });

  it('throws / errors without a Redux Provider', () => {
    // useSelector outside a Provider throws in React 18.
    expect(() => {
      render(<conversationsPlugin.effect! />);
    }).toThrow();  // or check consoleError was called
  });

  it('renders successfully inside a Redux Provider', () => {
    const store = configureStore({
      reducer: {
        conversation: (state = {convoCreationMessages: []}) => state,
      },
    });
    // Should not throw.
    render(
      <Provider store={store}>
        <conversationsPlugin.effect! />
      </Provider>,
    );
  });
});
```

This test serves as **living documentation**: any future plugin that uses
`useSelector` must be wrapped in the same `Provider` pattern, and the test makes
that requirement explicit.

### 5c. Structural placement test (optional integration test)

**Goal:** Catch "PluginEffects placed outside the Provider" regressions. This is
harder to test at the unit level because it requires rendering the actual GoB
`WebEffects.tsx` component tree. Two options:

**Option A — Smoke test in `dev/` folder (inside GoB):** Write a test in
`dev/WebEffects.test.tsx` inside the worktree. This test lives in a `.patch`
file: `patches/plugins_web_effects_test.patch`. The patch adds the test file and
the include glob. The downside is that the patch adds a test whose source isn't
in the source tree, creating a drift risk.

**Option B — Contract test on `PluginEffects` return type:** Statically verify
that `WebEffects` always returns `<PluginEffects>` by scanning its import graph.
This is a simpler approach that doesn't require a new patch, but it only catches
a missing import, not a wrong insertion point.

**Recommendation:** Start with Option A for maximum protection, but only add
that patch once the baseline (§5a and §5b) is proven working.

### 5d. Wire logic tests (`plugins-core/wire.test.ts`)

**Goal:** Verify `wirePlugins` pushes contributions into the featureManager
correctly, and is idempotent-safe.

```ts
import {describe, it, expect, beforeEach} from 'vitest';
import {wirePlugins} from './wire';
import {PluginRegistry} from './registry';
import type {Plugin, PluginFeatureManager} from './wire';

describe('wirePlugins', () => {
  let fm: PluginFeatureManager;

  beforeEach(() => {
    // Reset PluginRegistry state between tests (it's a module-level singleton).
    PluginRegistry.auxSideBarPanes.length = 0;
    PluginRegistry.sidebarSections.length = 0;
    PluginRegistry.conversationActions.length = 0;
    PluginRegistry.topLevelBarItems.length = 0;
    fm = {customAuxPanes: new Map(), sidebarItemsFeature: {items: []}};
  });

  it('wires a sidebar plugin into sidebarItemsFeature', () => {
    const SidebarComp = () => null;
    const plugin: Plugin = {
      id: 'test',
      sidebar: {label: 'Test', icon: 'star', component: SidebarComp},
    };
    wirePlugins(fm, [plugin]);
    expect(fm.sidebarItemsFeature!.items).toHaveLength(1);
    expect(fm.sidebarItemsFeature!.items[0].id).toBe('test');
  });

  it('wires an auxPane into featureManager.customAuxPanes', () => {
    const descriptor = {} as any;
    const plugin: Plugin = {id: 'p', auxPane: {id: 'myPane', descriptor}};
    wirePlugins(fm, [plugin]);
    expect(fm.customAuxPanes!.get('myPane')).toBe(descriptor);
  });

  it('is a no-op for plugins with no contributions', () => {
    wirePlugins(fm, [{id: 'empty'}]);
    expect(fm.sidebarItemsFeature!.items).toHaveLength(0);
  });
});
```

--------------------------------------------------------------------------------

## 6. Mocking strategy for Redux-dependent plugins

Plugins using `useSelector` need a Redux store. The recommended pattern is a
**minimal stub store** (not the real Antigravity store) created with
`configureStore`:

```tsx
// test-utils/makeStore.tsx  (lives in v2/plugins-core/ or v2/plugins/)
import {configureStore} from '@reduxjs/toolkit';

/** Builds a minimal Redux store for plugin tests. */
export function makeTestStore(preloadedState: Record<string, unknown> = {}) {
  return configureStore({
    reducer: Object.fromEntries(
      Object.entries(preloadedState).map(([key, val]) => [
        key,
        (state = val) => state,
      ]),
    ),
    preloadedState,
  });
}

/** Renders children inside a test Redux Provider. */
export function renderWithStore(
  ui: React.ReactElement,
  preloadedState: Record<string, unknown> = {},
) {
  const store = makeTestStore(preloadedState);
  return render(<Provider store={store}>{ui}</Provider>);
}
```

This avoids importing the real Antigravity Redux state (which has dozens of slices
and circular deps) while still satisfying `react-redux`'s context requirement.

--------------------------------------------------------------------------------

## 7. Running tests

### During development (inside a built worktree)

```bash
# Navigate to the active worktree (created by `./agent_manager.py build --build-frontend`)
cd ~/.agent_manager/worktrees/<workspace>/exa/agent_ui_toolkit

# Run all tests including plugin tests
npm run test

# Watch mode for TDD
npm run test:watch -- --reporter=verbose
```

### Via the CLI (proposed `./agent_manager.py test` integration)

The `agent_manager.py test` command (already mentioned in `README.md`) should:
1. Resolve the current worktree path. 2. `cd` into `exa/agent_ui_toolkit/`. 3.
Run `npm run test`.

This is a single-liner addition to `cli/builder.py`.

--------------------------------------------------------------------------------

## 8. Patch plan summary

| Patch file                               | What it does                      |
| ---------------------------------------- | --------------------------------- |
| `patches/plugins_vitest.patch.tpl`       | Adds `@plugins` + `@plugins-core` |
:                                          : aliases to `vitest.config.ts` and :
:                                          : extends `include` to source tree  :
:                                          : plugin dirs                       :
| *(no new patch)*                         | Test files live in source tree    |
:                                          : next to source — included via     :
:                                          : `include` glob                    :
| `patches/plugins_web_effects_test.patch` | *(Optional phase 2)* Adds         |
:                                          : structural smoke test for         :
:                                          : `WebEffects.tsx`                  :

--------------------------------------------------------------------------------

## 9. Open questions

1.  **Absolute source path in `include` glob**: Vitest's `include` supports
    absolute globs, but this creates a coupling between the worktree path and
    the source tree path. The builder already knows both — `__PLUGINS_DIR__` and
    `__PLUGINS_CORE_DIR__` are safe substitution targets. However, the glob must
    also work at runtime inside the worktree. **Validation needed:** try the
    glob in an existing worktree before writing the patch.

2.  **PluginRegistry singleton reset**: `PluginRegistry` is a module-level
    singleton (`const PluginRegistry = { ... }`). Tests that call `wirePlugins`
    will accumulate state across tests unless explicitly reset. Need to either
    expose a `reset()` function or restructure tests to be order-independent.

3.  **`conversationSlice` import**: The `ConversationRecorder` test needs to
    import or stub the real Redux slice that provides
    `state.conversation.convoCreationMessages`. The real slice likely lives deep
    in the GoB repo. Using a stub reducer avoids the import but hides future
    selector-path breakage. A dedicated slice mock is safer long-term.

4.  **Error boundary interference**: `PluginEffects` is supposed to wrap errors
    in an error boundary. The "crash without Provider" test (§5b) may not throw
    at the `render()` call level if the error boundary silently catches the
    error. Need to confirm: does React 18 + react-redux throw synchronously or
    only log via `console.error`? The test should be written accordingly.

--------------------------------------------------------------------------------

## 10. Recommended implementation order

1.  ✅ **Validate** that `npm run test` works in the current worktree as-is.
2.  **Write `wire.test.ts`** (pure logic, no React, no Redux — easiest).
3.  **Write `PluginEffects.test.tsx`** (React, no Redux — verifies isolation).
4.  **Write `plugin.test.tsx` for `conversations`** (React + Redux stub).
5.  **Write `patches/plugins_vitest.patch.tpl`** to extend the include glob and
    add aliases.
6.  **Rebuild worktree** and run `npm run test` to confirm all tests pass.
7.  **(Optional)** Add structural `WebEffects.test.tsx` via a second patch.
8.  **Update `agent_manager.py test`** to run `npm run test` in the worktree.
