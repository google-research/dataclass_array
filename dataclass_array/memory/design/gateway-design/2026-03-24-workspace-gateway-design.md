# Design: Workspace Gateway for Parallel Feature Testing

## Goal
Enable parallel testing of features (Frontend, Sidecar, and Language Server) developed in separate CitC workspaces through a single UI interface with seamless switching, maintaining isolation while allowing recovery from HEAD.

---

## 1. High-Level System Anatomy

The Agent Manager stack is divided into four main architectural pillars:

| Component | Path | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | `v1/scripts/` | `run_agent_manager.py` manages process lifecycles. `rebuild.sh` manages patch triggers. |
| **Gateway Sidecar** | `v1/sidecar/gateway/` | Pure infrastructure router. **Always alive** on primary port `3001`. |
| **Workspace Sidecar** | `v1/sidecar/workspace/` | Feature backend. Runs only when target workspace triggers backend edits. |
| **Logical Plugins** | `v1/plugins/` | Modular enhancements (Frontend + Backend) injected into the frame. |
| **Frontend Patches** | `v1/web/` | Diff records (`.patch`) applied to upstream Exafunction bundle source. |

---

## 2. Codebase Factorization & Code-Level Plugins

To ensure absolute separation between Gateway routing and Workspace features, while preventing process proliferation:

### A. `v1/sidecar/gateway/` (Individual process)
*   **`main.go`**: Starts the reverse proxy.
*   **`discovery.go`**: Scans environment to list switchable targets.
*   **`proxy.go`**: Implements dynamic `httputil.ReverseProxy` based on workspace cookie.

### B. `v1/sidecar/workspace/` (Single backend process)
Instead of setting up independent processes/ports for each plugin component, we use **Code-Level Modular Separation**:
*   Plugin backends live in: `v1/plugins/<plugin-name>/backend/`
*   Workspace `main.go` imports and registers these modular handlers during initialization.
*   **Benefit**: Keeps folder structure modular (agent conflict-free) while keeping the operational footprint lightweight (single executable context).

---

## 3. Workspace Discovery & Lazy Loading

*   **Registry**: `gateway/` reads switchable targets from scanned CitC lists.
*   **Single Active Component**: Gateway ensures that only **one** workspace backend stack is running at any point to save resources.
*   **Switch Overlay**: User experience includes a graceful loading overlay during the startup window required for Language Server initialization.

---

## 4. Dimensional Routing & Automatic Edit Detection

Edits trigger isolation strictly on modification detection relative to target workspace files.

### Automatic Edit Detection Logic
Before launching Workspace `X`, the Gateway evaluates `hg status` triggers:

*   **Rule 1: Frontend-only feature** (No edits in `v1/sidecar/` or standard LS paths)
    *   Gateway loads structure for statics (`dist/` via Isolate Worktree).
    *   Gateway routes `/api/*` requests to the **Default HEAD Sidecar**.
*   **Rule 2: Backend feature** (Edits detected in Sidecar or Plugins triggers package build)
    *   Gateway triggers isolated workspace Sidecar run on a dynamic port.
    *   Gateway routes `/api/*` requests to that Workspace Sidecar.

---

## 5. Frontend Isolation via `git worktree`

Currently, `rebuild.sh` patches a single absolute Git workspace, bottlenecking parallel edits.

**Solution**:
`rebuild.sh` will check out a dedicated **Git Worktree** for the workspace.
*   **Target**: `dev-worktree/<workspace-name>`
*   **Command**: `git worktree add <target> HEAD --detach`
*   Allows isolated frontend compilation supporting concurrent patch iteration.

---

## 6. UI Switcher Plugin

A new plugin inside the interface itself makes deployment seamless.

1.  **Discovery Endpoint**: Gateway serves `/gateway/workspaces` listing active CitC paths.
2.  **UI Selector Component**: In sidebar or auxiliary layout.
3.  **Mechanism**: Sets `x-agent-workspace=<name>` cookie and reloads.

---

## 7. Edge Cases & Graceful Recovery

*   **Visual Barrier**: The Gateway serves the **Switcher UI** independently (via internal route `/gateway/ui`), guaranteed live.
*   **Router Failover**: Proxy failures display a visual block page guiding user to **Retarget HEAD** in single click.

---

## 8. Design Considerations & FAQ (Preserving Reasoning)

*   **Why do we need separate port mapping for backends?**
    *   Multiple Sidecar/LS instances running concurrently on the same machine cannot bind to the exact same port due to standard `EADDRINUSE` blocks. Even with Lazy-Loading, dynamic ports prevent conflicts with remaining hanging connections.
*   **How to prevent race conditions in Workspace Discovery?**
    *   Instead of reading/writing to a single central `.json` file (which hits lock contention), each session writes independent node references in `~/.agent_manager/workspaces/<workspace>.json` with atomicity creation layers.
*   **Why are we using `git worktree`?**
    *   It completely side-steps single-tenant parallel patch deployment clashes seen on standard absolute references attached to directories.
*   **Can Frontend edits share underlying Backends?**
    *   Yes. With Dimensional Routing, if `hg status` reflects no backend triggers, Gateway preserves standard HEAD APIs completely bypassing auxiliary processes.
