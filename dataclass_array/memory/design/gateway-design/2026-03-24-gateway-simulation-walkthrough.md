# Simulation: Clean State Workspace Activation Timeline

This scenario simulates launching 3 separate features from a clean state (nothing built, nothing running) on a developer machine.

---

## 🌍 Initial State
*   Gateway Controller: **OFFLINE**
*   Shared dist Static assets: **EMPTY**
*   CITC Workspaces present:
    1.  `feat-a` (Edits in Sidebar UI + Sidecar Backend)
    2.  `feat-b` (Edits in Todo UI plugin only - *Frontend only*)
    3.  `feat-c` (Edits in Language Server configurations)

---

## 🗓️ Timeline Execution

### 📍 Step 1: Start Infrastructure
Developer boots the global Gateway router in their primary clone index.

*   **Command**: `./agent_manager.py start --gateway`
*   **Behind the Scenes Trigger**:
    1.  Compiles thin `sidecar/gateway/` binary executable context.
    2.  Starts Gateway reverse proxy server binding strictly to port **`3001`**.
*   **System State**: Gateway Dashboard is live at `http://localhost:3001/gateway/`.

---

### 📍 Step 2: Developer Accesses `feat-a`
Developer navigates to `http://localhost:3001/feat-a/` inside the browser.

*   **Action 1: Detection & Isolated Build**
    1.  Gateway receives request, scans location resolving offline target status.
    2.  Detects backend edits (`hg status` hits in Sidecar paths).
    3.  **Command Run**: `builder.py` triggers `git worktree add dev-worktree/feat-a HEAD --detach` creating isolate volume.
    4.  **Build**: Executes `npm install` and `vite build` inside isolated workspace dist, PLUS compiles Workspace-Sidecar binary.
*   **Action 2: Startup**
    1.  Gateway launches Workspace-Sidecar + Language Server on dynamic ports (e.g., `3012` & `3013`).
    2.  Gateway strips `/feat-a/` prefix and proxies continuous API buffers transparently.

---

### 📍 Step 3: Developer Switches to `feat-b` (Frontend-Only Edition)
Developer navigates to `http://localhost:3001/feat-b/`.

*   **Action 1: State Cleanup & Reaping**
    1.  Gateway recognizes new workspace switch overlay triggers.
    2.  Gateway **terminates previously running `feat-a` process backends** to prevent machine overload saturations.
*   **Action 2: Detection & Build**
    1.  Evaluates `hg status` for `feat-b` $\rightarrow$ **No Backend edits found**.
    2.  Builds *only* the frontend Static bundle inside isolated `dev-worktree/feat-b` statics isolate.
*   **Action 3: Transparent forwards**
    1.  Gateway serves static dist direct isolate accurately.
    2.  Forwards backend `/api/*` endpoints seamlessly to the **HEAD default Sidecar** avoiding process proliferation executables.

---

### 📍 Step 4: Developer Accesses `feat-c`
Developer navigates to `http://localhost:3001/feat-c/`.

*   Processes repeat similarly to **Step 2** (reap B, isolating full backend build triggers, spawn dynamic hooks). Memory saturation stays completely flat because parallel deployment loads lazy-reclaim consecutively flawless.
