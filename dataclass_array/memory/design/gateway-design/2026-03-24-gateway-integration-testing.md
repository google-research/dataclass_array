# Design: Automated Integration & E2E Testing via Gateway

## Goal
Enable concurrent, failure-isolated automated integration and End-to-End (E2E) tests for N parallel workspace branches without colliding environment address maps.

---

## 1. The Deterministic URL Leverage

Because every workspace adopts an explicit absolute sub-directory path structure (e.g., `/feat-x/`), test runner target endpoints become **100% predictable**.

A headless browser runner (like Playwright or Cypress) can specify isolated frame anchors directly in its base configuration configuration targets:

```typescript
// example E2E test config (e.g., Playwright)
export default defineConfig({
  use: {
    // Directs the tests to the isolated workspace environment transparently
    baseURL: 'http://localhost:3001/workspace-name/',
  },
});
```

---

## 2. Parallel execution benefits

### A. Zero Collision Pipelines
Multiple continuous integration workflows targeting **different branches** can interface against the core Gateway concurrently.
*   Runner A targets: `http://localhost:3001/feat-a/`
*   Runner B targets: `http://localhost:3001/feat-b/`
*   **Benefit**: Zero port contention lists block presubmits concurrently addressing alternate environments.

### B. Dynamic Stack Activation (Lazy Loader triggers)
Automatic goroutine checkpoints applied inside lazy-load handlers activate automatically even on automated clients:
1.  Test suite sends initial `GET /workspace-name/` request.
2.  Gateway wakes underlying isolated process logic.
3.  Runner successfully inspects visual indices loaded on isolated paths accurately.

---

## 3. Testing Checklist triggers via CLI

We can extend the orchestrator command lists (e.g., `agent_manager.py`) to support test validation phases natively:

`agent_manager.py test <workspace-name>`
*   **Step 1**: Ensures target isolated builders are compiled and live.
*   **Step 2**: Invokes runner setup appended targeting relevant folder directory anchor addresses.
