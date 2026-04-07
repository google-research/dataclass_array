---
name: notion
description: >-
  Read and query Notion databases and pages via the Notion API CLI. Use when
  you need to read Notion data.
---

# Notion CLI

Interact with Notion databases and pages using the Python CLI at:
`configs/users/epot/_agents/skills/notion/scripts/notion_cli.py`

**No dependencies** — uses only Python stdlib (urllib).

**API key** is read automatically from `~/.notion-keys/todo-read-only.txt`.
NEVER read or print the key — the CLI handles it internally.

## Usage

```bash
python configs/users/epot/_agents/skills/notion/scripts/notion_cli.py \
  [--api-key-file PATH] <command> [args...]
```

For brevity, set an alias:

```bash
alias notion="python /path/to/notion_cli.py"
```

## Commands

### `query-database` — list database entries

```bash
notion query-database <database_id>
notion query-database <database_id> --limit 10
notion query-database <database_id> --filter '{"property":"Status","select":{"equals":"Done"}}'
notion query-database <database_id> --sorts '[{"property":"Name","direction":"ascending"}]'
```

`database_id` accepts the full Notion URL, the bare UUID, or the 32-char hex.

### `get-database` — retrieve schema

```bash
notion get-database <database_id>
```

Returns the database title, description, and property schema — useful for
discovering property names before writing `--filter` expressions.

### `get-page` — retrieve page properties

```bash
notion get-page <page_id>
```

Returns all properties of a page (title, status, dates, etc.), but not the
body content. Use `get-blocks` for body text.

### `get-blocks` — read page body

```bash
notion get-blocks <page_id>
notion get-blocks <page_id> --recursive   # expand nested blocks
```

Returns the block tree of a page (paragraphs, headings, bullets, etc.).

### `search` — find pages and databases

```bash
notion search                          # all accessible objects
notion search "my query"
notion search --filter-type page       # only pages
notion search --filter-type database   # only databases
```

### `tree` — show hierarchical view of a database

Prints all entries as an indented tree. Every node shows its **full page ID**
so you can drill in immediately with `get-blocks`.

`--root` accepts a **name substring** or an **ID** (full UUID, raw 32-char hex,
or partial UUID substring). Both are matched case-insensitively.

```bash
# Full tree
notion tree 32c978a1730680d78f16f485603ab46f

# Root categories only (no children)
notion tree 32c978a1730680d78f16f485603ab46f --depth 1

# Subtree by name (substring match)
notion tree 32c978a1730680d78f16f485603ab46f --root agent_manager

# Subtree by UUID
notion tree 32c978a1730680d78f16f485603ab46f --root 32c978a1-7306-806c-a881-de021b492a12

# One level deep
notion tree 32c978a1730680d78f16f485603ab46f --root agent_manager --depth 2
```

Example output:

```
agent_manager  (32c978a1-7306-806c-a881-de021b492a12)
  - todo_view  (32c978a1-7306-80f4-8226-da0dcfa333b4)
    - Button to spin up agent to tackle the TODO  (32c978a1-7306-806d-b96b-d974863fbf12)
    - History of all previous attempt  (32c978a1-7306-80be-839b-f6755f413203)
  - cron_job triggers  (32c978a1-7306-80ec-b4dd-dea5339a2462)
```

> [!NOTE]
> The Notion API does not expose `has_children` in database query responses —
> pages come back as page objects, not block objects. There is no free signal
> for block content. To check if a node has a description, just call
> `get-blocks <id>` on the nodes you care about.

To read the body of any node:

```bash
notion get-blocks <id>            # flat list of blocks
notion get-blocks <id> --recursive  # expand nested blocks
```

## Known databases

| Database | ID |
|---|---|
| epot todos ("Agent TODOs") | `32c978a1730680d78f16f485603ab46f` |

### Listing todos by category (single call)

To list all todos under a named category, use `tree --root`:

```bash
python configs/users/epot/_agents/skills/notion/scripts/notion_cli.py \
  tree 32c978a1730680d78f16f485603ab46f --root <category_name>
```

Example — agent\_manager todos:

```bash
python configs/users/epot/_agents/skills/notion/scripts/notion_cli.py \
  tree 32c978a1730680d78f16f485603ab46f --root agent_manager
```

## Working with results

All commands output JSON to stdout. Use `jq` to extract fields:

```bash
# List all page titles
notion query-database 32c978a1730680d78f16f485603ab46f \
  | jq -r '.[].properties.Name.title[0].plain_text'

# Extract a specific property
notion query-database <id> | jq '.[].properties.Status.select.name'
```

## Tips

> [!TIP]
> Run `get-database <id>` first to discover property names, then use them in
> `--filter` expressions. Property names are case-sensitive.

> [!NOTE]
> Pagination is handled automatically — all results are returned unless
> `--limit` is set.
