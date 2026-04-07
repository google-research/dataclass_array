---
name: typescript
description: >
  Write TypeScript for browser-facing projects in the monorepo.
  Use when creating, converting, or building TypeScript that runs in the browser.
---

## Build rules

### tsconfig.json

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "bundler",
    "strict": true,
    "outDir": "./js",
    "rootDir": "./ts",
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"]
  },
  "include": ["ts/**/*.ts"]
}
```

### Running tsc locally (for dev server)

```bash
NODE=$(find /usr/local/google/home -name node -path "*Antigravity*" | head -1)
TSC=$(find /usr/local/google/home -name tsc -path "*typescript*" | head -1)
PATH=$(dirname $NODE):$PATH $TSC
```

`node` and `npx` are NOT on the default gLinux PATH. Find them in
`.Antigravity-server/`, `.bun/install/cache/`, or `.vscode-server/`.

## Style rules (Google TS style guide)

1.  **`const` over `let`; never `var`.**
2.  **ES modules only.** `import`/`export`; never globals or `<script>` per
    file.
3.  **No `any`.** Use `unknown` + narrowing, or a concrete type.
4.  **`interface` over `type` for object shapes.** `type` for unions/aliases.
5.  **PascalCase** for interfaces, types, enums. **camelCase** for functions,
    variables, properties.
6.  **`readonly` on data-only interfaces.** Mutable state belongs in explicit
    state objects, not data interfaces.
7.  **Throw `Error` objects**, not strings.
8.  **`catch (e: unknown)`** with `e instanceof Error` narrowing.
9.  **`dataset`** for DOM data storage, never expando properties.
10. **Omit JSDoc when the type signature is sufficient.** Add JSDoc only for
    non-obvious behavior.

## CDN library declarations

For libraries loaded via `<script>` tags (marked, highlight.js, DOMPurify),
create `externals.d.ts` with `declare namespace` blocks. Check for `typeof`
before use so the code gracefully degrades if the CDN fails.
