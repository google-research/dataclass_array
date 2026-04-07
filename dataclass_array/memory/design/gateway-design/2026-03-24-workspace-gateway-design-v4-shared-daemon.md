# Design v4: Shared Daemon Workspace Gateway

## 1. Overview
The **Shared Daemon Architecture** streamlines multi-tenant workspace orchestration by sharing core background executables (Language Server & default Sidecar) for all routing buffers. It leverages dynamic RPC offset payload triggers (like `cwd`) to eliminate duplicate backing process proliferation while scale branches isolates safely.

---

## 2. Component Map & Lifecycle

| Component | Instance Count | Lifecycle | Purpose |
| :--- | :--- | :--- | :--- |
| **Gateway Proxy** | **1** | Permanent | Port `3001`. Custom multiplexes prefixes routing traffic securely. |
| **HEAD Sidecar** | **1** | Permanent | Port `3000`. Responds to default index `/api/` routing payloads for nodes with **no local Back-End edits**. |
| **HEAD Language Server** | **1** | Permanent | Port `8081`. Executes all Agent command triggers (`runCommand`, `searchCode`) utilizing dynamic CWD routing anchors. |
| **Workspace-Specific Sidecar** | **$0..N$** | On-Demand | **Only spawned** if developer makes explicit Go-layer edits addressing local patch overlays. |

---

## 3. Dimensional Routing Map (Target Workspace `feat-x`)

### A. Front-End Statics (`/head/` vs `/feat-x/`)
1.  Front-End is compiled strictly into isolated folders offsets (e.g. `dev-worktree/feat-x/dist/`).
2.  Injects relative `base: './'` parameter offsets inside `vite.config.ts` deployments to ensure asset addresses remain location-agnostic anchors.

### B. Backend RPCs Node multiplexing

When the browser interacts with Gateway Port `3001`, the logic parses transparent offsets:

```go
// Gateway Routing Decision Matrix
func routeEndpoints(request Request) string {
    if hasGoEdits("workspace-a") {
        return "http://localhost:3012" // Isolated workspace sidecar
    }
    return "http://localhost:3000"     // Default HEAD Sidecar Share node
}
```

**For Agent Execution hooks** (`RunCommand`):
The front-end encapsulates absolute triggers injecting local worktree setups contextually:
```typescript
lsClient.runCommand({
    command: "hg status",
    cwd: "/google/src/cloud/.../isolated-worktrees/workspace-a/"
})
```

---

## 4. Operational scaling benefits

1.  **Flat Memory Saturation**: Maximum executing processes stay static, saving CPU bundles.
2.  **Zero Startup Overhead streams delay**: You don't need process lazy-load buffering offsets waiting for multi-server handshakes consecutive execution cycles.
3.  **Flat setup bounds**: Safe concurrency mapped transparently.
