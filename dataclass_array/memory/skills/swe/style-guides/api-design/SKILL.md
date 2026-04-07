---
name: api-design
description: >
  Architecture, abstraction design, and module-boundary rules.
  Read BEFORE designing any new feature.
---

# API and Abstraction Design

> For the high-level feature design *process* (user experience first,
> orthogonality checks), see `feature-design`.

## Code should match high-level semantics

-   Each line of business logic should map to one clear, high-level action (e.g.
    `launch_evaluations()`, `wait_for_results()`, `write_result_to_db(result)`).
-   If a block of code requires a comment to explain what it does (not why),
    extract it into a named function or method that makes the intent
    self-evident.
-   Never mix levels of abstraction in the same function: a function that calls
    `launch_evaluations()` should not also contain low-level socket or
    serialization code.
-   **Pass dependencies, don't reach for globals.** A function should receive
    its inputs as parameters — not access module-level state (flags, singletons)
    internally. Access global state at the top-level entry point and propagate.
-   **Question the function's existence.** When a function mixes unrelated
    concerns, the fix may be to inline it and let the caller compose the pieces
    — not to clean up its internals.
-   **One name per concept.** The same concept must use the same name everywhere
    — across functions, files, and modules. If a value is called `origin` in one
    place and `provenance` in another, pick one.
-   **Backward compatibility.** When changing an API, consider existing callers,
    configs, and notebooks. Provide a migration path if the change is breaking.

## Layered architecture

Organize code in clear dependency layers. Lower layers are general-purpose
utilities; upper layers compose them into domain-specific abstractions.

```
Public API (__init__.py re-exports)
  └── Domain objects (e.g. dataclass-based value types)
        └── Operations / algorithms (standalone functions operating on domain objects)
              └── Utilities (pure helpers: math, numpy wrappers, string manipulation)
```

Each layer only depends on the layer below it.

## Class abstractions

-   Prefer **frozen dataclasses** as the primary data abstraction. Never mutate
    — always produce new instances with `self.replace(field=new_value)`.
-   Use `__post_init__` to normalize/validate inputs. This is the only place
    where mutation is allowed with `object.__setattr__(self, ...)`.
-   Prefer `@functools.cached_property` to methods. Complex logic can often be
    written as cached_property calling other cached_property.
-   When construction requires non-trivial logic or there are multiple ways to
    create an object, provide named `@classmethod` factories (`from_X`,
    `from_Y`) instead of overloading `__init__`.
-   Define extensibility points as **protocols** (expected method names).

## Code organization

### One feature per file

-   Complex new features should go in a **separate new file**, not be appended
    to an existing utility module. Keep files focused on a single
    responsibility.
-   Self-contained subsystems (parsers, grammars, formatters, serializers) must
    be planned as **separate modules from the start**, not as inline code in the
    calling module.

### `__init__.py` and public API

-   The top-level `__init__.py` defines the **public API surface** through
    explicit re-exports.
-   Group exports by category with short comments.

### Module boundaries

-   **Logic belongs with the data it operates on.** When adding a new function,
    place it in the module that owns the data structure it reads or mutates —
    not in the caller that happens to need it first.
-   **Cross-module → public.** If a function is called from another module, it
    must be public (no leading `_`). Private functions are module-internal only.
