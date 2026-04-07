# Namespaces

A **namespace** is a URL-based runtime environment with its own compiled frontend
(built from a `git worktree` of the GoB repo), pointed at by the gateway via the
registry. Each namespace is accessible at `http://<host>:3001/<name>/`.

## When do you need a namespace?

Use a namespace when you want to test a feature branch **simultaneously** with
another branch, in a separate browser tab, without restarting the server.

If you only need to test one branch at a time, just use `/head/` — no namespace
setup required.

## Step-by-step: adding a namespace

### 1. Create and switch to your feature branch

```bash
# In your CitC client or git repo:
hg new-bookmark feat-x
# ... make your changes ...
```

### 2. Build the workspace

To create a new namespace entry in the registry (e.g. `feat-x`), you just run
the build:

```bash
./agent_manager.py build --build-frontend
```

> **Tip:** In a CitC workspace, you can use `--build-frontend` to automatically resolve the namespace name to your current CitC client name.

This does three things:
1. Creates a `git worktree` at `~/.agent_manager/worktrees/feat-x`
2. Runs `npm install && npm run build` inside it (requires `base: './'` in `vite.config.ts`)
3. Writes a registry entry at `~/.agent_manager/namespaces/feat-x.json`

The gateway picks up the new entry within 2 seconds (TTL cache). No restart needed.

### 3. Open in browser

```
http://<your-hostname>:3001/feat-x/
```

You can now have `/head/` and `/feat-x/` open in separate tabs, each showing the
live state of their respective branch.

### 4. Rebuild after frontend changes

If you've changed TypeScript/CSS:

To rebuild it, just run again:

```bash
./agent_manager.py build --build-frontend
```

If `node_modules` already exists, this skips `npm install` and only runs `vite build`.

Force a full reinstall:

```bash
./agent_manager.py build --build-frontend --force
```

## Registry entry format

The build command writes `~/.agent_manager/namespaces/feat-x.json`:

```json
{
  "name": "feat-x",
  "path": "/home/user/.agent_manager/worktrees/feat-x",
  "sidecar_port": 3000,
  "ls_port": 3002,
  "has_go_edits": false,
  "bundle_path": "/home/user/.agent_manager/worktrees/feat-x/dist"
}
```

The gateway reads this file and routes `/feat-x/*` accordingly:
- `/feat-x/api/*` → sidecar on `sidecar_port`
- `/feat-x/*` → language server on `ls_port`

## Go edits and isolated sidecars

If your namespace requires an isolated sidecar due to Go-layer edits in its CitC workspace, the build
command sets `has_go_edits: true`. This is detected via `hg status sidecar/` in the CitC workspace.

> **Note:** Isolated per-namespace sidecars are not yet implemented. Currently all
> namespaces share the HEAD sidecar on port 3000, regardless of `has_go_edits`.
> This is the next planned feature in v2.

## Removing a namespace

Delete the registry file:

```bash
rm ~/.agent_manager/namespaces/feat-x.json
```

The gateway will return 404 for `/feat-x/*` within 2 seconds. You can also clean
up the worktree:

```bash
git worktree remove ~/.agent_manager/worktrees/feat-x
```

## Frontend requirement: relative asset paths

The gateway strips the namespace prefix before forwarding requests. For example,
`/feat-x/assets/main.js` becomes `/assets/main.js` before hitting the LS.

This means the frontend **must** be built with relative asset paths. In
`vite.config.ts`:

```typescript
export default defineConfig({
  base: './',   // Required — do NOT use '/' or an absolute URL
  // ...
})
```

If assets use absolute paths (e.g. `src="/assets/main.js"`), they will try to load
from `http://host:3001/assets/main.js`, bypassing the workspace prefix. The root
isolation middleware will block these as a 400 error.

## Viewing registered namespaces

```bash
./agent_manager.py status
```

Or hit the gateway API directly:

```bash
curl http://localhost:3001/api/gateway/namespaces
```

Returns a JSON array of all registered `NamespaceEntry` objects.
