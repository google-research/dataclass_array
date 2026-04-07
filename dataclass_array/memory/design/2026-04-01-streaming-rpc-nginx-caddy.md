# Streaming RPC Issues & nginx/Caddy Evaluation

**Date:** 2026-04-01
**Context:** `JetboxSubscribeToState` gRPC-web streaming hangs; evaluating whether to replace the custom Go gateway with nginx or Caddy.

---

## What's actually broken with streaming today

The streaming hang on `JetboxSubscribeToState` has **three distinct root causes**, all in the current custom Go gateway.

### 1. `ModifyResponse` buffers the entire response body

In `proxy.go`, `newLSProxy` reads the full body into memory to inject `<base>` tags:

```go
body, err := io.ReadAll(resp.Body)   // ← blocks until EOF
```

For regular HTML this is fine. But `ModifyResponse` is called on **every response**
routed to the LS — including gRPC-web streaming responses with
`Content-Type: application/grpc-web+proto`. `io.ReadAll` won't return until the
stream closes, turning an infinite server-sent stream into a hung request.

The current code does skip non-`text/html` content types:

```go
if !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/html") {
    return nil
}
```

So this *shouldn't* be the culprit for gRPC-web under normal conditions. But the
condition is fragile: if the LS sends `text/html` for a redirect before the stream,
or if `Content-Type` arrives late/wrong, `ModifyResponse` swallows the stream.

### 2. `logResponse` middleware wraps with a non-flushing `ResponseWriter`

In `main.go`, every request goes through `logResponse`, which wraps the real
`ResponseWriter` with `loggingResponseWriter`:

```go
type loggingResponseWriter struct {
    http.ResponseWriter
    statusCode int
}
```

This struct embeds `http.ResponseWriter` but **does not implement `http.Flusher`**.

For streaming responses, the LS writes data and flushes; but the proxy's call to
`w.(http.Flusher).Flush()` type-asserts to `Flusher` on the *wrapper* — which
fails silently or panics depending on Go's type system. Without flushing, data
accumulates in the OS TCP buffer and the client sees nothing until the connection
closes.

### 3. `serveFromReferer` routing: cross-request state is lost for plugin-internal calls

The referer fallback works for the first request but it's slow and fragile — each
streaming chunk that loses its namespace prefix triggers a 404 → referer-extract →
re-proxy cycle. For gRPC-web where the browser multiplexes over a single long-lived
connection, losing the namespace prefix mid-stream is fatal.

---

## Would nginx or Caddy fix these?

**Yes — but only the proxy-level issues.** Here's the comparison:

| Issue | Custom Go gateway | nginx | Caddy |
|---|---|---|---|
| `ModifyResponse` buffering streaming | ❌ Requires careful fix | ✅ Transparent passthrough | ✅ Transparent passthrough |
| Non-flushing response wrapper | ❌ Requires fix | ✅ Non-issue | ✅ Non-issue |
| WebSocket upgrades | ✅ Works (httputil handles it) | ✅ `proxy_http_version 1.1` | ✅ Native |
| gRPC-web passthrough | ❌ Fragile | ⚠️ Needs `grpc_pass` or tuning | ✅ Native `reverse_proxy` streams |
| Namespace path-prefix routing | ✅ Native | ✅ `location /ns/ {}` blocks | ✅ `handle /ns/* {}` |
| HTML injection (`<base>` tag) | ✅ Custom Go code | ❌ Not possible without lua/filter | ❌ Not built-in |
| Supervisor (spawn/restart LS+sidecar) | ✅ Built-in | ❌ Not a process manager | ❌ Not a process manager |
| Referer-based routing fallback | ✅ Custom Go code | ❌ Limited | ❌ Limited |
| Registry/namespace TTL cache | ✅ Built-in | ❌ Needs reload/templating | ❌ Needs Caddy API |

**The big blocker**: nginx and Caddy are static-config proxies. The current gateway
does two things that aren't in their wheelhouse:

1. **HTML injection** — modifying `index.html` to inject `<base href="/{namespace}/">`
   and `window.__NAMESPACE__`. This is what makes the multi-namespace SPA work.
2. **Dynamic namespace discovery** — reading a JSON registry file and spawning new
   child processes on-demand.

Caddy has an API for live reloading config, and nginx can do upstreams, but neither
gives you the child-process lifecycle management currently provided by `supervisor.go`.

---

## Recommendation: fix the Go gateway, don't replace it

The actual streaming fix is **targeted and small**. Three changes:

### Fix 1 — Make `loggingResponseWriter` implement `http.Flusher`

```go
func (lrw *loggingResponseWriter) Flush() {
    if f, ok := lrw.ResponseWriter.(http.Flusher); ok {
        f.Flush()
    }
}
```

### Fix 2 — Guard `ModifyResponse` more strictly for streaming content types

In `newLSProxy`, skip buffering for any streaming content type:

```go
proxy.ModifyResponse = func(resp *http.Response) error {
    ct := resp.Header.Get("Content-Type")
    // Skip buffering for streaming content types.
    if strings.Contains(ct, "grpc") || strings.Contains(ct, "event-stream") {
        return nil
    }
    if !strings.HasPrefix(ct, "text/html") {
        return nil
    }
    // ... existing HTML injection ...
}
```

### Fix 3 — Set `FlushInterval = -1` on the LS proxy

`FlushInterval = -1` (available since Go 1.20) tells `httputil.ReverseProxy` to
copy each chunk immediately as it arrives rather than accumulating:

```go
proxy := httputil.NewSingleHostReverseProxy(target)
proxy.FlushInterval = -1  // flush immediately on each write
proxy.Transport = &http.Transport{
    DisableCompression: true,  // prevents buffering in deflate/gzip readers
}
```

---

## When nginx/Caddy *would* make sense

If the HTML injection feature were removed (e.g. by configuring the SPA to read its
base path from a cookie or by serving a separate `config.json`), the gateway would
become a pure path-routing reverse proxy. At that point nginx/Caddy would be
genuinely simpler to maintain than custom Go.

Caddy in particular would be appealing because:
- Its `reverse_proxy` directive handles WebSockets and streaming natively.
- The Caddy API (`:2019`) lets you dynamically add/remove upstreams without restart.
- No need to write `FlushInterval` hacks.

But even then, something would need to spawn/reap the LS and sidecar processes —
likely `systemd --user`, `supervisord`, or a stripped-down Go supervisor alongside
Caddy.

---

## Summary

| | Replace with nginx/Caddy | Fix Go gateway |
|---|---|---|
| Fixes streaming | ✅ | ✅ |
| Keeps HTML injection | ❌ (would need to remove feature) | ✅ |
| Keeps dynamic namespace routing | ❌ (partial via Caddy API) | ✅ |
| Keeps child-process supervisor | ❌ (need separate tool) | ✅ |
| Scope of work | Large refactor | ~20 lines of Go |

**Fix `FlushInterval`, `Flusher`, and the `ModifyResponse` guard in the Go gateway.**
That is the minimal fix for the streaming hang. Revisit nginx/Caddy if the gateway
ever grows complex enough that maintaining it feels like a burden — right now it is
~400 lines and does exactly what it needs to.
