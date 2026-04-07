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

"""List metadata from all memory fragments in a directory tree."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---", re.DOTALL)
_KEY_RE = re.compile(r"^(\w[\w-]*):\s*(.*)")


def parse_frontmatter(text: str) -> dict[str, str] | None:
  """Extract YAML frontmatter key-value pairs from markdown text.

  Handles flat ``key: value`` pairs and multi-line values (e.g. ``key: >``
  followed by indented continuation lines).

  Returns:
    A dict of metadata, or ``None`` when no frontmatter block is found.

  Example:
    ```python
    meta = parse_frontmatter("---\\ntitle: Hello\\n---\\nBody.")
    assert meta == {"title": "Hello"}
    ```
  """
  match = _FRONTMATTER_RE.search(text)
  if not match:
    return None

  meta: dict[str, str] = {}
  current_key: str | None = None
  parts: list[str] = []

  def _flush() -> None:
    nonlocal current_key, parts
    if current_key is None:
      return
    # Drop the YAML block-scalar indicator (> or |) if it's the first part.
    if parts and parts[0] in (">", "|"):
      parts = parts[1:]
    val = " ".join(parts).strip()
    if val:
      meta[current_key] = val
    current_key = None
    parts = []

  for line in match.group(1).splitlines():
    m = _KEY_RE.match(line)
    if m:
      _flush()
      current_key = m.group(1)
      parts = [m.group(2).strip()]
    elif current_key is not None and line.startswith(("  ", "\t")):
      # Continuation line for multi-line value.
      parts.append(line.strip())
  _flush()

  return meta or None


# ---------------------------------------------------------------------------
# Fragment reading
# ---------------------------------------------------------------------------


def scan_fragments(
    root: str,
    *,
    type_filter: str | None = None,
    status_filter: str | None = None,
    author_filter: str | None = None,
    max_depth: int | None = None,
) -> list[dict[str, str]]:
  """Walk *root* and return metadata dicts for every ``.md`` file found.

  Files without parseable frontmatter are still included with
  ``_status: no-metadata`` so that missing or malformed fragments are visible
  rather than silently dropped.

  When *max_depth* caps traversal, subdirectories at the boundary are emitted
  as ``_status: directory`` entries so they remain visible.

  Args:
    root: Directory to scan.
    type_filter: If set, only include fragments whose ``type`` matches.
    status_filter: If set, only include fragments whose ``status`` matches.
    max_depth: Maximum directory depth to recurse into.  ``0`` scans only *root*
      itself, ``1`` includes its immediate subdirectories, and so on. ``None``
      (the default) means unlimited depth.
  """
  fragments: list[dict[str, str]] = []
  root = os.path.normpath(root)
  root_depth = root.rstrip(os.sep).count(os.sep)

  for dirpath, dirnames, filenames in os.walk(root):
    current_depth = dirpath.rstrip(os.sep).count(os.sep) - root_depth

    # When we've reached the depth limit, list subdirectories as entries and
    # prevent os.walk from descending further.
    if max_depth is not None and current_depth >= max_depth:
      for dname in sorted(dirnames):
        dpath = os.path.join(dirpath, dname)
        relpath = os.path.relpath(dpath, root)
        fragments.append({"_path": relpath + os.sep, "_status": "directory"})
      dirnames.clear()

    for fname in sorted(filenames):
      if not fname.endswith(".md"):
        continue
      path = os.path.join(dirpath, fname)
      relpath = os.path.relpath(path, root)
      try:
        with open(path, encoding="utf-8") as f:
          text = f.read(4096)  # Frontmatter is always near the top.
      except OSError:
        fragments.append({"_path": relpath, "_status": "read-error"})
        continue
      meta = parse_frontmatter(text)
      if meta is None:
        fragments.append({"_path": relpath, "_status": "no-metadata"})
        continue
      meta["_path"] = relpath
      if type_filter and meta.get("type", "") != type_filter:
        continue
      if status_filter and meta.get("status", "") != status_filter:
        continue
      if author_filter and meta.get("author", "") != author_filter:
        continue
      fragments.append(meta)
  return fragments


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_blocks(fragments: list[dict[str, str]]) -> str:
  """Render fragments as readable ``key: value`` blocks, one per fragment."""
  if not fragments:
    return "(no fragments found)"
  blocks: list[str] = []
  for f in fragments:
    path = f.get("_path", "?")
    lines = [f"── {path}"]
    for k, v in f.items():
      if k == "_path":
        continue
      lines.append(f"   {k}: {v}")
    blocks.append("\n".join(lines))
  return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_recurse(value: str) -> int | None:
  """Convert a ``--recurse`` CLI value to an int or ``None`` (unlimited)."""
  if value == "*":
    return None
  try:
    depth = int(value)
  except ValueError:
    raise argparse.ArgumentTypeError(
        f"Invalid recurse depth: '{value}'. Use 0, 1, 2, ... or '*'."
    )
  if depth < 0:
    raise argparse.ArgumentTypeError(
        f"Recurse depth must be non-negative, got {depth}."
    )
  return depth


def main() -> None:
  """Entry point for the ``list_fragments`` CLI."""
  parser = argparse.ArgumentParser(
      description="List metadata from memory fragments.",
  )
  parser.add_argument("directory", help="Root directory to scan.")
  parser.add_argument(
      "--type",
      dest="type_filter",
      default=None,
      help="Filter by fragment type (e.g. todo, episode).",
  )
  parser.add_argument(
      "--status",
      dest="status_filter",
      default=None,
      help="Filter by status (e.g. active, blocked).",
  )
  parser.add_argument(
      "--recurse",
      default="0",
      metavar="DEPTH",
      help=(
          "Nesting depth: 0 = root only, 1 = one level of subdirs, "
          "* = unlimited (default: *)."
      ),
  )
  parser.add_argument(
      "--json",
      dest="as_json",
      action="store_true",
      help="Output as JSON instead of a table.",
  )
  parser.add_argument(
      "--author",
      dest="author_filter",
      default=None,
      help="Filter by author (e.g. user, agent).",
  )
  args = parser.parse_args()

  if not os.path.isdir(args.directory):
    print(
        f"Error: '{args.directory}' is not a directory.",
        file=sys.stderr,
    )
    sys.exit(1)

  max_depth = _parse_recurse(args.recurse)
  fragments = scan_fragments(
      args.directory,
      type_filter=args.type_filter,
      status_filter=args.status_filter,
      author_filter=args.author_filter,
      max_depth=max_depth,
  )

  if args.as_json:
    print(json.dumps(fragments, indent=2))
  else:
    print(format_blocks(fragments))
    print(f"\n({len(fragments)} fragment(s) found)")


if __name__ == "__main__":
  main()
