# Specification: Unified Orchestrator CLI (`agent_manager.py`)

## 1. Overview
The `agent_manager.py` script replaces `run_agent_manager.py` and `rebuild.sh`. It acts as a single entrypoint supporting isolated builds, background process orchestration (Gateway vs Workspace streams), and live process diagnostics.

---

## 2. Subcommands & Flag Matrix

### 🚀 Command: **`start`**
Launches backing core permanent infrastructure servers safely wrapped in `tmux` execution isolates (detaches and exits immediately).

*   **Sub-parms**: None required. Boots Gateway (`3001`), Default Sidecar (`3000`), & Shared LS (`8081`) simultaneously.

---

### 🛑 Command: **`stop`**
Safely reaps isolated executing process buckets nodes.

| Flag | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| **`--all`** | *Flag* | Aggressive cleaner reaps **every** associated tmux session safely. | `true` |

---

### 🛠️ Command: **`build`**
Compiles isolated statics dist templates inside underlying isolating Worktree streams.

| Flag | Type | Description | Default |
| :--- | :--- | :--- | :--- |
| **`--workspace [NAME]`** | *String* | **(Required)** Target name specifying isolating output isolates. | None |
| **`--force`** | *Flag* | Ignores local edit detection overlays forcing clean `npm install` builds. | `false` |

> [!NOTE]
> **Implicit Isolate Boot**: If the builder detects custom **Go local edits** sitting inside target paths, the `build` command automatically kicks off the isolate Workspace Sidecar supervisor loop immediately following compile consecutive offsets safely.

---

### 📊 Command: **`status`**
Displays the current executing stack registry table.

*   Reads `.agent_manager/workspaces/` isolates.
*   **Output example**:
    ```text
    Gateway: RUNNING (Port 3001, session: gateway_sidecar)

    Workspaces:
    - Head:         OFFLINE
    - feat-x:       RUNNING (Sidecar: 3015, LS: 3016, session: fix-todo-stack)
    - fix-bug:      OFFLINE
    ```

---

## 3. Logging & State Node buffers

*   **Process Daemon management**: Execution forces continuous backgrounding wrapper nodes scoped by naming session isolates (e.g., `tmux attach -t <workspace-name>`).
*   **Execution Log endpoints**: Output buffers stream automatically targeting agnostic cache resolvers:
    `~/.agent_manager/logs/<workspace-name>_sidecar.log`
