---
name: feature-design
description: >
  Process for designing complex features end-to-end. Read when scoping a new
  feature request, before any implementation planning or code design.
---

# Feature Design Process

You **MUST** add those steps explicitly to your `Task` plan.

-   [ ] Design <feature name> (feature-design process)
    -   [ ] Step 1: Write end-user experience
    -   [ ] Step 2: Iterate on the design
    -   [ ] Step 3: Write API design plan — get user approval

## Step 1 — Write the end-user experience

Write the exact command, config line, or function call the user will type. If
you can't state the user experience in one concrete example, you don't yet
understand the feature.

## Step 2 — Iterate on the design

Look at each of those points, and try to see whether the original user
experience could be improved.

-   **Concrete** — there is at least one complete, copy-pasteable example.
-   **Orthogonal** — the feature composes with existing ones without
    cross-product explosion. If adding it requires N copies of every existing
    variant (configs, subclasses, function overloads), the abstraction boundary
    is wrong. It should be an independent axis — a flag, a parameter, a mixin.
-   **No duplicated inputs** — every piece of information the user provides is
    specified exactly once. If the same semantic value (e.g. TPU platform, model
    name, dataset path) appears in two different config surfaces (flags, config
    objects, environment), the abstraction is leaking. Derive the duplicate from
    the single source. **Procedure:** for each field in the proposed API, ask:
    "does this information already exist somewhere the user already provides
    it?" List every field and its existing source (or "new"). If any field has
    an existing source, derive it — don't ask the user to provide it twice.
-   **Reuse existing surfaces** — prefer reusing existing flags, config fields,
    and CLI patterns over introducing new ones. If the information already has a
    natural home, use it instead of creating a parallel config path.
-   **Minimal** — no unnecessary concepts or steps are introduced.

Do 5 rounds of improvements.

## Step 3 — Write the API design plan

Define the public API surface: exported names, their signatures, and what they
mean. Include the user experience from step 1. Present this to the user for
review **before** starting the implementation plan.

Only after user approval, proceed to the implementation plan and hand off
internal architecture decisions to `api-design`.

> **IMPORTANT**: Do NOT think about *how* to implement the feature until the
> API design is approved. Steps 1–3 are purely about the user-facing surface.
> Implementation research happens only after approval.

## Related

-   `api-design` — internal architecture after the public API is approved
