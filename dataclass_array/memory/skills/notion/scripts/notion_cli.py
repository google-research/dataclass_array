# Copyright 2026 The dataclass_array Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Notion CLI — query databases, read pages, and search via the Notion REST API.

Uses only Python stdlib (urllib) — no extra dependencies required.

Usage:
  python notion_cli.py [--api-key-file PATH] <command> [args...]

Commands:
  query-database <database_id> [--filter JSON] [--sorts JSON] [--limit N]
  get-database   <database_id>
  get-page       <page_id>
  get-blocks     <block_id> [--recursive]
  search         [query] [--filter-type page|database]

The API key is read from --api-key-file
(default: ~/.notion-keys/todo-read-only.txt).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_DEFAULT_KEY_FILE = str(
    Path.home() / ".notion-keys" / "todo-read-only.txt"
)
_API_BASE = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _load_api_key(path: str) -> str:
  """Read and strip the Notion API key from *path*."""
  key_path = Path(path).expanduser()
  if not key_path.exists():
    print(f"ERROR: API key file not found: {key_path}", file=sys.stderr)
    sys.exit(1)
  return key_path.read_text().strip()


def _request(
    api_key: str,
    method: str,
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
  """Make an authenticated Notion API request, return parsed JSON."""
  url = f"{_API_BASE}/{endpoint}"
  if params:
    query = urllib.parse.urlencode(params)
    url = f"{url}?{query}"

  data = json.dumps(body).encode() if body else None
  req = urllib.request.Request(url, data=data, method=method)
  req.add_header("Authorization", f"Bearer {api_key}")
  req.add_header("Notion-Version", _NOTION_VERSION)
  req.add_header("Content-Type", "application/json")

  try:
    with urllib.request.urlopen(req) as resp:
      return json.loads(resp.read())
  except urllib.error.HTTPError as e:
    body_text = e.read().decode()
    print(f"ERROR {e.code}: {e.reason}\n{body_text}", file=sys.stderr)
    sys.exit(1)




def _paginate(
    api_key: str,
    method: str,
    endpoint: str,
    body: dict[str, Any] | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
  """Collect all pages of a paginated Notion endpoint."""
  results: list[dict[str, Any]] = []
  cursor: str | None = None

  while True:
    payload = dict(body or {})
    if cursor:
      payload["start_cursor"] = cursor
    if limit and not cursor:
      payload["page_size"] = min(limit, 100)

    response = _request(api_key, method, endpoint, body=payload or None)
    batch = response.get("results", [])
    results.extend(batch)

    if limit and len(results) >= limit:
      return results[:limit]

    if not response.get("has_more"):
      break
    cursor = response.get("next_cursor")

  return results


def _dump(obj: object) -> None:
  """Pretty-print *obj* as JSON to stdout."""
  print(json.dumps(obj, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_query_database(api_key: str, args: argparse.Namespace) -> None:
  """Query all entries from a Notion database."""
  body: dict[str, Any] = {}
  if args.filter:
    body["filter"] = json.loads(args.filter)
  if args.sorts:
    body["sorts"] = json.loads(args.sorts)

  limit = args.limit or 0
  results = _paginate(
      api_key,
      "POST",
      f"databases/{args.database_id}/query",
      body=body or None,
      limit=limit,
  )
  _dump(results)
  print(f"\n# {len(results)} entries", file=sys.stderr)


def cmd_get_database(api_key: str, args: argparse.Namespace) -> None:
  """Retrieve database schema and metadata."""
  result = _request(api_key, "GET", f"databases/{args.database_id}")
  _dump(result)


def cmd_get_page(api_key: str, args: argparse.Namespace) -> None:
  """Retrieve a page's properties by page_id."""
  result = _request(api_key, "GET", f"pages/{args.page_id}")
  _dump(result)


def cmd_get_blocks(api_key: str, args: argparse.Namespace) -> None:
  """List all child blocks of a page or block."""
  blocks = _paginate(
      api_key, "GET", f"blocks/{args.block_id}/children"
  )
  if args.recursive:
    blocks = _expand_blocks(api_key, blocks)
  _dump(blocks)


def _expand_blocks(api_key: str, blocks: list[dict]) -> list[dict]:
  """Recursively expand children of blocks that have_children."""
  for block in blocks:
    if block.get("has_children"):
      children = _paginate(
          api_key, "GET", f"blocks/{block['id']}/children"
      )
      block["children"] = _expand_blocks(api_key, children)
  return blocks


def cmd_search(api_key: str, args: argparse.Namespace) -> None:
  """Search pages and databases."""
  body: dict[str, Any] = {}
  if args.query:
    body["query"] = args.query
  if args.filter_type:
    body["filter"] = {"property": "object", "value": args.filter_type}

  results = _paginate(api_key, "POST", "search", body=body or None)
  _dump(results)
  print(f"\n# {len(results)} results", file=sys.stderr)


def cmd_tree(api_key: str, args: argparse.Namespace) -> None:
  """Print all database entries as an indented tree.

  Each entry shows its name and page ID.
  Use the ID with get-blocks or get-page to explore further.

  Note: the Notion API does not expose whether a page has block content in
  database query responses (pages are returned as page objects, not block
  objects, so has_children is absent). To check content, call get-blocks
  on the specific IDs you are interested in.

  Example::

    notion tree 32c978a1730680d78f16f485603ab46f --root agent_manager
  """
  entries = _paginate(
      api_key,
      "POST",
      f"databases/{args.database_id}/query",
  )

  # Build lookup tables.
  id_to_name: dict[str, str] = {}
  id_to_children: dict[str, list[str]] = {}
  id_to_parents: dict[str, list[str]] = {}

  for entry in entries:
    eid = entry["id"]
    title = entry.get("properties", {}).get("Name", {}).get("title", [])
    id_to_name[eid] = title[0].get("plain_text", "(no name)") if title else "(no name)"
    id_to_children[eid] = [
        r["id"] for r in entry.get("properties", {}).get("Sub-item", {}).get("relation", [])
    ]
    id_to_parents[eid] = [
        r["id"] for r in entry.get("properties", {}).get("Parent item", {}).get("relation", [])
    ]

  def _print_node(eid: str, depth: int = 0) -> None:
    prefix = "  " * depth + ("- " if depth > 0 else "")
    name = id_to_name.get(eid, eid)
    print(f"{prefix}{name}  ({eid})")
    if args.depth == 0 or depth < args.depth - 1:
      for child_id in id_to_children.get(eid, []):
        _print_node(child_id, depth + 1)

  if args.root:
    needle = _normalise_id(args.root)  # no-op if plain text, UUID if ID
    matches = [
        eid for eid, name in id_to_name.items()
        if needle.lower() in name.lower() or needle.lower() in eid.lower()
    ]
    if not matches:
      print(f"ERROR: no entry matching '{args.root}'", file=sys.stderr)
      sys.exit(1)
    for eid in matches:
      _print_node(eid)
  else:
    roots = [eid for eid, parents in id_to_parents.items() if not parents]
    for eid in roots:
      _print_node(eid)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
  """Build the argument parser."""
  parser = argparse.ArgumentParser(
      description="Notion API CLI — no extra dependencies required.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      "--api-key-file",
      default=_DEFAULT_KEY_FILE,
      help="Path to file containing the Notion integration token.",
  )
  sub = parser.add_subparsers(dest="command", required=True)

  # query-database
  p_query = sub.add_parser(
      "query-database", help="Query all entries in a database."
  )
  p_query.add_argument("database_id")
  p_query.add_argument(
      "--filter",
      metavar="JSON",
      help=(
          "Notion filter object as JSON, e.g."
          ' \'{"property":"Status","select":{"equals":"Done"}}\''
      ),
  )
  p_query.add_argument(
      "--sorts",
      metavar="JSON",
      help=(
          "Notion sorts array as JSON, e.g."
          ' \'[{"property":"Name","direction":"ascending"}]\''
      ),
  )
  p_query.add_argument(
      "--limit",
      type=int,
      default=0,
      help="Max results (0 = all).",
  )

  # get-database
  p_db = sub.add_parser("get-database", help="Retrieve database schema.")
  p_db.add_argument("database_id")

  # get-page
  p_page = sub.add_parser("get-page", help="Retrieve page properties.")
  p_page.add_argument("page_id")

  # get-blocks
  p_blocks = sub.add_parser(
      "get-blocks", help="List child blocks of a page or block."
  )
  p_blocks.add_argument("block_id")
  p_blocks.add_argument(
      "--recursive",
      action="store_true",
      help="Recursively expand nested blocks.",
  )

  # search
  p_search = sub.add_parser("search", help="Search pages and databases.")
  p_search.add_argument("query", nargs="?", default="", help="Search query.")
  p_search.add_argument(
      "--filter-type",
      choices=["page", "database"],
      help="Restrict to pages or databases only.",
  )

  # tree
  p_tree = sub.add_parser(
      "tree",
      help="Print database entries as an indented hierarchy with IDs.",
  )
  p_tree.add_argument("database_id", help="Database ID or URL.")
  p_tree.add_argument(
      "--root",
      metavar="NAME",
      default="",
      help="Show only the subtree whose name contains NAME (case-insensitive).",
  )
  p_tree.add_argument(
      "--depth",
      type=int,
      default=0,
      metavar="N",
      help="Max depth to display (1 = root only, 2 = root + children, 0 = unlimited).",
  )

  return parser


def _normalise_id(raw: str) -> str:
  """Strip URL noise and return the bare UUID."""
  # Handle full Notion URLs like https://www.notion.so/user/32c978a1...?v=...
  raw = raw.split("?")[0].split("/")[-1]
  raw = raw.replace("-", "")
  if len(raw) == 32:
    return f"{raw[:8]}-{raw[8:12]}-{raw[12:16]}-{raw[16:20]}-{raw[20:]}"
  return raw


def main() -> None:
  """Entry point."""
  parser = _build_parser()
  args = parser.parse_args()

  api_key = _load_api_key(args.api_key_file)

  # Normalise any ID arguments.
  for attr in ("database_id", "page_id", "block_id"):
    if hasattr(args, attr):
      setattr(args, attr, _normalise_id(getattr(args, attr)))

  dispatch = {
      "query-database": cmd_query_database,
      "get-database": cmd_get_database,
      "get-page": cmd_get_page,
      "get-blocks": cmd_get_blocks,
      "search": cmd_search,
      "tree": cmd_tree,
  }
  dispatch[args.command](api_key, args)


if __name__ == "__main__":
  main()
