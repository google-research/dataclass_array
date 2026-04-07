# Namespace Routing & Redirection Flows

This document describes how the Namespace Gateway handles routing and redirection
for different namespaces (e.g., `head` or a dynamic namespace) and ensures assets
and API calls are correctly routed.

## High-Level Architecture

The Gateway acts as a reverse proxy, multiplexing requests between backend services
for a given namespace:

1. **Sidecar**: Handles namespace-specific API requests (`/api/*`).
2. **Language Server (LS)**: Serves the namespace dashboard, IDE interface, static
   assets, and long-lived streaming RPCs.

### Request Routing

| Flow Type | URL Pattern | Backend | Description |
| :--- | :--- | :--- | :--- |
| **Dashboard Load** | `/head/` or `/<ns>/` | Language Server | Serves the SPA `index.html`. |
| **Static Assets** | `/head/assets/*` | Language Server | Serves static files (JS, CSS, etc.). |
| **Sidecar APIs** | `/head/api/*` | Sidecar | Custom API endpoints. |
| **Streaming RPCs** | `/head/<rpc-path>` | Language Server | gRPC-web / SSE long-lived streams. |

---

## Implemented Systems

### 1. Direct Segment Routing

The Gateway inspects the first path segment of the URL path to determine the target
namespace:

- `/head/*` routes to the primary HEAD namespace.
- `/<namespace>/*` looks up `<namespace>` in the registry and routes to its ports.

### 2. Referer-Based Namespace Recovery

For requests that lack the namespace prefix (e.g., absolute fetch calls like
`/api/foo`), the Gateway falls back to inspecting the `Referer` header. It scans
the Referer URL for a known namespace name and routes the request accordingly,
preventing broken links for absolute resource loads.

### 3. Just-In-Time Script and Base Tag Injection

To enable the frontend SPA to correctly maintain its router base and resolve relative
assets, the Gateway intercepts HTML responses from the Language Server and injects a
`<base>` tag and a script snippet into the `<head>` tag:

```html
<base href="/<namespace_name>/">
<script>window.__NAMESPACE__ = "<namespace_name>";</script>
```

This allows:

1. **Relative Asset Loading**: All relative paths (e.g.,
   `<script src="./assets/index.js">`) resolve correctly relative to the namespace
   root, even when the current page is a deeply nested SPA route
   (e.g., `/head/c/<id>`). This fixes "blank screen on refresh" issues.
2. **Client-Side Router Base**: The client-side router reads this namespace name to
   prepend it to all subsequent navigations and requests.

The HTML injection is **skipped** for any response whose `Content-Type` contains
`grpc` or `event-stream` — see [Streaming RPC Passthrough](#4-streaming-rpc-passthrough)
below.

### 4. Streaming RPC Passthrough

The Language Server exposes long-lived streaming RPCs over **gRPC-web**
(e.g. `JetboxSubscribeToState`). These use
`Content-Type: application/grpc-web+proto` and require the proxy to forward each
chunk to the browser as it arrives, rather than buffering until the stream closes.

Three settings on the LS reverse proxy (`proxy.go: newLSProxy`) make this work:

| Setting | Value | Why |
| :--- | :--- | :--- |
| `FlushInterval` | `-1` | Tells `httputil.ReverseProxy` to write each chunk to the client immediately. The default (100 ms timer) causes visible lag or a hung stream on long-lived connections. |
| `Transport.DisableCompression` | `true` | Prevents the gzip transport reader from accumulating compressed chunks before handing them to the proxy writer. |
| `ModifyResponse` guard | skips `grpc` / `event-stream` content types | The HTML injection reads the full response body with `io.ReadAll`, which blocks until the stream closes. The guard ensures only `text/html` responses are ever buffered. |

The **logging middleware** (`main.go: loggingResponseWriter`) also implements
`http.Flusher` by delegating to the underlying `ResponseWriter`. Without this, the
`Flusher` interface is hidden when the middleware wraps the real writer, silently
breaking the flush path for all streaming responses.

```
Browser
  └──► logResponse wrapper       (implements http.Flusher)
         └──► namespaceProxy
                └──► httputil.ReverseProxy   (FlushInterval = -1)
                       └──► Language Server  (gRPC-web stream)
```

> **Rule**: never add buffering middleware between the gateway listener and the LS
> proxy without re-implementing `http.Flusher`. Any wrapper that drops `Flusher`
> will silently break all streaming RPCs.
