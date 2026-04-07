# Design v2: Workspace Gateway for Parallel Feature Testing

## Goal
Enable parallel testing of features (Frontend, Sidecar, and Language Server) developed in separate CitC workspaces through a single UI interface with seamless switching, maintaining absolute isolation while allowing accurate fallback recovery from HEAD.

---

## 1. High-Level System Anatomy

The Agent Manager stack is divided into four modular architectural pillars:

| Component | Path | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | `v1/scripts/` | `run_agent_manager.py` manages process lifecycles. `rebuild.sh` manages isolated patch triggers. |
| **Gateway Sidecar** | `v1/sidecar/gateway/` | Pure infrastructure router. **Always alive** on primary port `3001`. |
| **Workspace Sidecar** | `v1/sidecar/workspace/` | Feature backend. Combines core logic with Code-Level Plugins. Runs dynamically. |
| **Logical Plugins** | `v1/plugins/` | Modular enhancements (Frontend + Backend) injected into the modular frame. |
| **Frontend Patches** | `v1/web/` | Diff records (`.patch`) applied to upstream Exafunction bundle source. |

---

## 2. Architecture Diagram

```mermaid
graph TD
    subgraph "User Edge (Browser)"
        UI["Agent Manager UI"]
        WS["Workspace Switcher Plugin"]
    end

    subgraph "Central Router (%: Always Alive)"
        Gateway["Gateway Sidecar (Port 3001)"]
        Discovery["Discovery Scanner"]
        Gateway --> Discovery
    end

    subgraph "Workspace Stack: 'feature-x' (%: Lazy Loaded)"
        WorkspaceSidecar["Workspace Sidecar (Dynamic Port)"]
        LS["Language Server (Dynamic Port)"]
        Worktree["Git Worktree (Bundle dist/)"]
    end

    subgraph "Default HEAD (Fallback)"
        HeadSidecar["HEAD Sidecar"]
        HeadLS["HEAD Language Server"]
    end

    UI -->|HTTP / WebSockets| Gateway
    WS -->|Set Cookie 'x-agent-workspace'| Gateway

    Gateway -.->|Target: 'feature-x'| WorkspaceSidecar
    Gateway -.->|Target: 'feature-x'| LS
    Gateway -.->|Fallback| HeadSidecar
    Gateway -.->|Fallback| HeadLS

    WorkspaceSidecar -->|Serve Statics| Worktree
```

---

## 3. Data Decision Flow: Dimensional Routing

When a request hits the Gateway, it follows a contextual filter traversal to optimize loading speed:

```mermaid
graph TD
    Start["Request Received"] --> CheckCookie{"Has x-agent-workspace cookie?"}

    CheckCookie -->|No| RouteHead["Route to default HEAD stack"]

    CheckCookie -->|Yes| LoadTarget["Resolve Workspace from Isolate Registry"]
    LoadTarget --> DetectEdits{"Has Backend Edits? (hg status)"}

    DetectEdits -->|No: Frontend-only| ServeIsolateStatics["Serve dist/ from Worktree"]
    ServeIsolateStatics --> ProxyDefaultAPI["Proxy /api/* to Head Sidecar"]

    DetectEdits -->|Yes: Full Feature| ServeIsolateStatics2["Serve dist/ from Worktree"]
    ServeIsolateStatics2 --> SpawnStack["Spawn Workspace Sidecar + LS"]
    SpawnStack --> ProxyWorkAPI["Proxy /api/* to Workspace Sidecar"]
```

---

## 4. Core Implementation Pillars

### A. Code-Level Plugins (Modularity)
Instead of multiplying background routers for each feature, we use code-level isolation:
*   **Location**: `v1/plugins/<name>/backend/` handlers.
*   **Mechanic**: The `workspace/main.go` aggregates and loads these modular endpoints at boot.
*   **Result**: Single active process footer with completely decoupled authoring folders.

### B. Git Worktrees (Compile Safety)
To allow N workspaces to compile statics in parallel:
*   Trigger: `rebuild.sh` creates isolate nodes: `dev-worktree/<workspace-name>`
*   Command wrapper: `git worktree add <target> HEAD --detach`
*   Output isolates continuous compiles without stepping over absolute path pointers.

### C. Lazy Loading & Failure Failovers
*   Gateway enforces a **Single Active Component** trigger checklist to preserve machine TPU/Memory overhead filters.
*   **Automatic Failover**: Proxy failures display an auxiliary standalone Error Barrier with a standalone click back to safety node overlay.

---

## 5. Design Considerations & FAQ

*   **Why do we need dynamic port mapping?**
    *   Multiple Sidecar/LS builds running consecutively cannot execute against standard `EADDRINUSE` binds.
*   **How do we block lookup race conditions?**
    *   Nodes avoid singular json lock contention, instead favoring isolate file buffers created in `~/.agent_manager/workspaces/<workspace>.json`.
*   **How seamless is the UI layer?**
    *   Cookie assignment with standalone cache reloading transitions loaded frames directly without frame drop hooks.
