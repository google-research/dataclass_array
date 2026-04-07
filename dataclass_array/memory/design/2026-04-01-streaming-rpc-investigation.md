# Streaming RPC Investigation — Agent Manager v2

_Date: 2026-04-01_

## Summary

The frontend communicates with the Language Server (LS) using **gRPC-Web** via
the **ConnectRPC** framework (not raw gRPC over HTTP/2). All RPC calls,
including streaming ones, go through the **Go gateway** reverse proxy on
`:3001`. The gateway forwards them to the LS over plain HTTP/1.1.

---

## Full Call Stack

```
Browser (React)
    │  gRPC-Web (Connect protocol, HTTP/1.1, JSON format)
    │  POST /head/exa.language_server_pb.LanguageServerService/JetboxSubscribeToState
    ▼
Gateway  (:3001, plain HTTP/1.1)
    │  httputil.ReverseProxy  (Go stdlib, HTTP/1.1)
    │  Strips /head/ prefix → forwards to LS port
    ▼
Language Server  (:300X, plain HTTP/1.1)
    │  ConnectRPC handler (Go)
    │  JetboxSubscribeToState → server-streaming response
    ▼
Browser receives chunked stream
```

---

## Frontend Client Setup

### Library

| Package                  | Version | Role                                |
|--------------------------|---------|-------------------------------------|
| `@connectrpc/connect`    | `^2.0.0`| RPC client core                     |
| `@connectrpc/connect-web`| `^2.0.0`| gRPC-Web / Connect protocol adapter |
| `@bufbuild/protobuf`     | –       | Proto serialization                 |

### Transport Initialization

File: `exa/agent_ui_toolkit/src/clients/StandaloneAgentClient.ts`

```typescript
const transport = createGrpcWebTransport({
  baseUrl,               // see below
  interceptors: [csrfInterceptor],
  useBinaryFormat: false, // uses JSON (Content-Type: application/grpc-web+json)
});
this.lsClient = createClient(LanguageServerService, transport);
```

> **`useBinaryFormat: false`** → The wire format is `application/grpc-web+json`
> (not binary protobuf). Each RPC request is a standard HTTP POST.

### Base URL Computation

The `baseUrl` is constructed by the `client_config.patch` we inject:

```typescript
// Original upstream code:
let baseUrl = config.baseUrl || window.location.origin;

// Our patch appends the namespace prefix:
const namespace = (window as any).__NAMESPACE__;
if (namespace && !config.baseUrl) {
  baseUrl = `${baseUrl}/${namespace}`;
}
```

The `__NAMESPACE__` global is injected by the gateway's `proxy.go` into every
`index.html` response:

```go
script := fmt.Sprintf(`<script>window.__NAMESPACE__ = "%s";</script>`, namespace)
```

So the actual `baseUrl` seen by ConnectRPC is:

```
http://<host>:3001/head
```

### RPC URL Pattern

ConnectRPC constructs request URLs as:

```
POST {baseUrl}/{fully-qualified-service}/{method}
```

For `JetboxSubscribeToState`:

```
POST http://<host>:3001/head/exa.language_server_pb.LanguageServerService/JetboxSubscribeToState
```

The gateway strips `/head/` → forwards as:

```
POST http://localhost:<ls-port>/exa.language_server_pb.LanguageServerService/JetboxSubscribeToState
```

---

## Streaming RPC: `JetboxSubscribeToState`

### Proto definition

```proto
// In language_server.proto:
rpc JetboxSubscribeToState(JetboxSubscribeToStateRequest)
    returns (stream JetboxSubscribeToStateResponse) {}
```

This is a **server-streaming RPC**: one request, potentially infinite responses.

### Frontend call site

File: [`ls-debug/plugin.tsx`](file:///research-v2-streaming-rpc/agent_manager/v2/plugins/ls-debug/plugin.tsx)

```typescript
const stream = lsClient.jetboxSubscribeToState({}, { signal: abortCtrl.signal });
for await (const msg of stream) {
  // process first message, then break
}
```

The ConnectRPC library implements server-streaming over gRPC-Web using
**chunked HTTP/1.1** (or HTTP/2 if available). Since our gateway uses plain
`httputil.ReverseProxy` over HTTP/1.1, the stream arrives as a series of
length-prefixed frames in the response body.

### Backend handler

File: `third_party/Antigravity/language_server/rpcs_jetbox_state.go`

```go
func (s *Server) JetboxSubscribeToState(
    ctx context.Context,
    connectReq *connect.Request[...],
    stream *connect.ServerStream[...],
) error {
    updates := make(chan *jetbox_state_pb.State, 10)
    s.jetboxStateStore.Subscribe(ctx, func(state *jetbox_state_pb.State) {
        select {
        case updates <- state:
        case <-ctx.Done():
        }
    })
    for {
        select {
        case <-ctx.Done(): return ctx.Err()
        case state := <-updates:
            stream.Send(...)
        }
    }
}
```

The RPC blocks indefinitely until the context is cancelled (client disconnect)
or an error occurs. **The first message is only sent when
`jetboxStateStore.WriteState()` is called** — the stream hangs until there is
a state write.

---

## Language Server HTTP Server Configuration

File: `third_party/Antigravity/language_server/server.go`

The LS runs **two** HTTP servers:

| Server | Protocol    | Port     | Notes                              |
|--------|-------------|----------|------------------------------------|
| HTTPS  | HTTP/2 + TLS| separate | For desktop IDE (h2 + ALPN)        |
| HTTP   | HTTP/1.1    | `--http_server_port` (`:300X`) | For browser / gateway proxy |

Both servers set `SetUnencryptedHTTP2(true)` in `http.Protocols`, so they
*can* do h2c (cleartext HTTP/2). However, the gateway's
`httputil.ReverseProxy` connects over **plain HTTP/1.1** only (no h2c
negotiation), so in practice the gateway↔LS leg runs HTTP/1.1.

The LS uses the `connect-go` library (`connectrpc.com/connect`), which natively supports all three protocols:
- **Connect** (POST + JSON or binary, Content-Type: `application/connect+…`)
- **gRPC-Web** (Content-Type: `application/grpc-web+…`)
- **gRPC** (HTTP/2 only, Content-Type: `application/grpc`)

Since the browser transport uses `createGrpcWebTransport` with
`useBinaryFormat: false`, the wire Content-Type is:

```
Content-Type: application/grpc-web+json
```

---

## Gateway Proxy Behavior for Streaming

File: [`proxy.go`](file:///research-v2-streaming-rpc/agent_manager/v2/gateway/proxy.go)

```go
proxy := httputil.NewSingleHostReverseProxy(target)
// ModifyResponse only fires for non-stream, html responses
proxy.ModifyResponse = func(resp *http.Response) error {
    if !strings.HasPrefix(resp.Header.Get("Content-Type"), "text/html") {
        return nil  // gRPC-web responses pass through unmodified
    }
    // ... inject <base> tag and __NAMESPACE__ script
}
```

### Critical issue with streaming

`httputil.ReverseProxy` uses `http.ResponseWriter.Write()` to forward the
response body. For streaming gRPC-Web, the LS sends a chunked HTTP/1.1 body.
Go's `httputil.ReverseProxy`:

1. **Does not buffer** the response body — it copies directly.
2. **Does not flush** proactively — it relies on the underlying
   `http.ResponseWriter` to detect that the client is still connected.
3. Uses `io.Copy` internally, which reads in 32 KB chunks. For gRPC-Web frames
   that are much smaller, this can cause **buffering delays**.

The `ModifyResponse` hook **reads the entire body** into memory for HTML
responses. For non-HTML responses (gRPC-Web), it returns `nil` immediately, so
streaming is pass-through. However, **if the `ModifyResponse` function is
called on a streaming response with a body that never ends** (e.g., a
connection classified as HTML by mistake), it would hang waiting for
`io.ReadAll`.

### The `logResponse` middleware wrapper

```go
srv := &http.Server{
    Addr:    fmt.Sprintf(":%d", *port),
    Handler: logResponse(mux),  // wraps with a loggingResponseWriter
}
```

`logResponse` wraps the `ResponseWriter` in a `loggingResponseWriter` that
intercepts `WriteHeader`. This wrapper does NOT interfere with streaming body
writes.

---

## CSRF Token Flow

The LS is started with `--csrf_token <random-hex>` (generated per run in
`main.go:generateCSRFToken()`). The frontend reads this token from the injected
`<script>` block in `index.html` (served by the LS's `serveIndexWithConfig`).

The ConnectRPC CSRF interceptor attaches it as a header on every RPC:

```
x-codeium-csrf-token: <token>
```

The gateway passes this header through unmodified (it's a raw reverse proxy).
The LS's CSRF interceptor validates it.

> **The token is per-namespace** (regenerated each time the LS process
> starts). After a restart, the browser must reload to get the new token.

---

## Root Cause Summary for Streaming Hang

Based on the code, there are several potential reasons the
`JetboxSubscribeToState` stream would hang (show "Timed out after 5s"):

1. **No initial state write**: The server handler only sends a message when
   `jetboxStateStore.WriteState()` is called. If the LS starts fresh with an
   empty state store and no client writes state, the stream blocks indefinitely.

2. **Gateway buffering**: The Go `http.ResponseWriter` on the gateway side may
   buffer the response before flushing to the browser. Unless the LS sets
   `Transfer-Encoding: chunked` explicitly (which Connect-go does for
   streaming), the response won't be flushed until the buffer fills.

3. **Referer-based routing misfire**: If a RPC request arrives without a valid
   namespace prefix (e.g., a race condition during page load), the gateway's
   `serveFromReferer` fallback may fail to route it correctly, returning a 404.
   The browser-side ConnectRPC client would then receive a non-gRPC-web
   response and fail with a parse error, not a timeout.

4. **HTTP/1.1 vs gRPC-Web framing**: Since ConnectRPC's gRPC-Web transport
   uses chunked HTTP/1.1, the `Transfer-Encoding: chunked` header must be
   preserved end-to-end. Go's `httputil.ReverseProxy` handles this correctly
   in theory, but if the LS doesn't properly set trailers (which gRPC-Web
   uses to transmit the final status), the stream may terminate unexpectedly.

---

## Key Files Reference

| File | Role |
|------|------|
| `exa/agent_ui_toolkit/src/clients/StandaloneAgentClient.ts` | Creates gRPC-Web transport + typed LS client |
| `agent_manager/v2/patches/client_config.patch` | Injects `window.__NAMESPACE__` into baseUrl |
| `agent_manager/v2/gateway/proxy.go` | Gateway reverse proxy; injects `<base>` tag in HTML |
| `agent_manager/v2/gateway/server.go` | Gateway routing; Referer fallback |
| `agent_manager/v2/gateway/main.go` | LS startup flags (standalone, http_server_port, jetbox, etc.) |
| `third_party/Antigravity/language_server/server.go` | LS HTTP mux setup; `setUpHTTPServers` (HTTP/1.1 + TLS/h2) |
| `third_party/Antigravity/language_server/rpcs_jetbox_state.go` | `JetboxSubscribeToState` server handler |
| `exa/language_server_pb/language_server.proto` | Service definition (line 551: streaming RPC) |
| `agent_manager/v2/plugins/ls-debug/plugin.tsx` | Frontend streaming client call (line 336) |
