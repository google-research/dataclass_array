# Specification: Responsibility Separation Matrix (CLI vs Gateway)

## To ensure robust execution bounds and avoid HTTP timeout collisions flakily stalling buffer nodes, the Workspace Gateway system strictly distributes **Building** and **Routing** responsibilities.

## 🛠️ 1. The Python CLI orchestrator (`agent_manager.py`)

*   **Role**: **Build & Startup orchestrator**.
*   **Context**: Runs on-demand inside the terminal clone frames triggered by
    the **Developer**. ### Responsibilities:
*   **Gateway setup**: Compiles the absolute thin `sidecar/gateway/` Go binary
    context overlays.
*   **Process reaps**: Manages re-spawns rejoining TMUX background environments
    securely.
*   **Isolate compiling triggers (The Builder)**:
    *   Creates detached worktree structures.

    ## * Executes isolating static compiles triggers consecutive `npm install && vite build` bounds targeting isolated setups statics folders directly.

    ## 🌐 2. The Gateway Controller (`sidecar/gateway/`)
*   **Role**: **Stateless Forwarding Router / File Server**.
*   **Context**: Runs continuously addressing background server buffers
    consecutive framing safety. ### Responsibilities:
*   **Path Prefixes Multiplexing**: Strips `/workspace-a/` segment transparently
    mapping transparent loads upstream.
*   **Files Serving Directs**: Inspects static caches isolated overlays deployed
    by the builder isolate, rendering indexing assets seamlessly.
*   **Dimensional forwards**: Forwards backend `/api/*` requests to the shared
    default backends stack seamlessly without doubling process multiplication
    offsets. > [!IMPORTANT] \
    > **The Gateway is 100% Stateless**. It never invokes compiler streams,
    re-attaches bash subprocesses, or waits on slow 30-second `npm` compilations
    blocking browser threads flakily. --- ## 🗓️ Typical Developer Workflow
    Example | Developer Action | CLI Role (`agent_manager.py`) | Gateway Role
    (`sidecar/gateway/`) | | :--- | :--- | :--- | | **1. `start --gateway`** |
    Boots background tmux permanents nodes safely. | Stands up stateless
    listening port `3001`. | | **2. `build --workspace feat-x`** | Detaches
    Worktree, runs isolated compilation isolates populated into static volume
    caches. | **Idle / Uninvolved**. | | **3. Visit `http://epot:3001/feat-x/`**
    | **Idle / Uninvolved**. | Intercepts segment forwards static index buffers
    transparently flawless. |
