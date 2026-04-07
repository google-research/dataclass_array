---
name: create-cl
description: >
  How to create and update CLs: commit, upload, and format CL descriptions.
---

## Creating / updating a CL

1.  Include **all** changed files in the CL — code changes, BUILD updates, and
    any skill file updates from the self-improvement step.
2.  **Choose amend vs commit.** Use `hg commit` to create a new CL. Use `hg
    amend` only to update an **existing draft CL**.
    -   `hg amend` on a public changeset fails with `abort: cannot amend public
        changesets`.
    -   Never `hg amend` a submitted CL — check `hg ll -r .` first; if the
        output shows `<submitted>`, always use `hg commit` instead.
3.  Run `hg upload -r .` to upload the CL. **Never use `hg mail`** — just
    upload.
4.  Notify the user with the CL number so they can review / mail it themselves.
5.  If the CL already exists (e.g. iterating on review feedback), use `hg amend`
    then `hg upload -r .` to update it.

## CL description formatting

When creating CLs for `third_party/py/` projects, the description **must** use
one of these formats:

*   Multi-line:

    ```markdown
    BEGIN_PUBLIC
    External facing description of the change...
    END_PUBLIC

    (optional) Additional internal description...

    OPTIONA_TAG=value
    ```

*   Single-line:

    ```markdown
    PUBLIC: One-line description

    (optional) Additional internal description...

    OPTIONAL_TAG=value
    ```

The `BEGIN_PUBLIC`/`END_PUBLIC` block (or `PUBLIC:` line) contains **only** the
external-facing description. NOT any of:

*   Internal references (build system, monorepo paths, etc.)
*   Comments to the reviewer

Those can be added afterwards.

All tags (e.g. `TAG=agy`, etc.) always go at the end.

If the cl **only** contains internal changes, format should be:

```markdown
Internal description...

PUBLIC: Internal

OPTIONAL_TAG=value
```

## Guideline

Descriptions text should follow those rules:

*   Structure: **one short tl;dr line**, then optionally a blank line followed
    by additional context.
*   Keep it **compact** — give high-level context, not a file-by-file
    enumeration. The diff already shows which files changed. Prefer compact
    bullet points.
*   ** **Prefer the simplest format.** When the public and internal descriptions
    are identical (i.e. there's nothing internal-only to say), use `PUBLIC:` —
    never wrap the same text in `BEGIN_PUBLIC`/`END_PUBLIC`.
