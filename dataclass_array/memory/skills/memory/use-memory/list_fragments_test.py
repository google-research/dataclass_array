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

from __future__ import annotations

import argparse
import importlib.util
import os

import pytest

# The skill directory has hyphens, so we import by file path.
_SCRIPT = os.path.join(os.path.dirname(__file__), "list_fragments.py")
_spec = importlib.util.spec_from_file_location("list_fragments", _SCRIPT)
list_fragments = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(list_fragments)


# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param(
            "---\ntitle: Hello\ntype: todo\n---\nBody.",
            {"title": "Hello", "type": "todo"},
            id="simple_key_value",
        ),
        pytest.param(
            "---\ndescription: >\n  First line\n  second line.\n---\n",
            {"description": "First line second line."},
            id="multiline_folded",
        ),
        pytest.param(
            "---\ndescription: |\n  Line one\n  Line two\n---\n",
            {"description": "Line one Line two"},
            id="multiline_literal",
        ),
        pytest.param(
            "---\nblocked-on: something\n---\n",
            {"blocked-on": "something"},
            id="hyphenated_key",
        ),
        pytest.param(
            (
                "---\n"
                "name: my-skill\n"
                "description: >\n"
                "  A long\n"
                "  description.\n"
                "type: reference\n"
                "---\n"
            ),
            {
                "name": "my-skill",
                "description": "A long description.",
                "type": "reference",
            },
            id="mixed_single_and_multiline",
        ),
        pytest.param(
            "---\ntitle: Has trailing space   \n---\n",
            {"title": "Has trailing space"},
            id="trailing_whitespace_stripped",
        ),
    ],
)
def test_parse_frontmatter(text, expected):
  assert list_fragments.parse_frontmatter(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("No frontmatter at all.", id="no_delimiters"),
        pytest.param("---\n\n---\n", id="empty_block"),
        pytest.param("---\n---\n", id="zero_width_block"),
    ],
)
def test_parse_frontmatter_returns_none(text):
  assert list_fragments.parse_frontmatter(text) is None


# ---------------------------------------------------------------------------
# scan_fragments
# ---------------------------------------------------------------------------


@pytest.fixture
def fragment_tree(tmp_path):
  """Return ``(root, write)`` where ``write(relpath, content)`` creates files."""

  def _write(relpath: str, content: str):
    path = tmp_path / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)

  return tmp_path, _write


def test_finds_fragments_recursively(fragment_tree):
  root, write = fragment_tree
  write("a.md", "---\ntitle: A\ntype: episode\n---\n")
  write("sub/b.md", "---\ntitle: B\ntype: todo\n---\n")

  result = list_fragments.scan_fragments(str(root))

  with_title = [f for f in result if "title" in f]
  assert len(with_title) == 2
  assert {f["title"] for f in with_title} == {"A", "B"}


def test_skips_non_md_files(fragment_tree):
  root, write = fragment_tree
  write("readme.txt", "---\ntitle: Nope\n---\n")
  assert list_fragments.scan_fragments(str(root)) == []


def test_shows_files_without_frontmatter(fragment_tree):
  root, write = fragment_tree
  write("plain.md", "# Just a heading\nNo frontmatter here.")

  result = list_fragments.scan_fragments(str(root))

  assert len(result) == 1
  assert result[0]["_status"] == "no-metadata"
  assert result[0]["_path"] == "plain.md"


def test_type_filter(fragment_tree):
  root, write = fragment_tree
  write("a.md", "---\ntitle: A\ntype: episode\n---\n")
  write("b.md", "---\ntitle: B\ntype: todo\n---\n")

  result = list_fragments.scan_fragments(str(root), type_filter="todo")

  with_title = [f for f in result if "title" in f]
  assert len(with_title) == 1
  assert with_title[0]["title"] == "B"


def test_status_filter(fragment_tree):
  root, write = fragment_tree
  write("a.md", "---\ntitle: A\ntype: todo\nstatus: active\n---\n")
  write("b.md", "---\ntitle: B\ntype: todo\nstatus: blocked\n---\n")

  result = list_fragments.scan_fragments(str(root), status_filter="active")

  with_title = [f for f in result if "title" in f]
  assert len(with_title) == 1
  assert with_title[0]["title"] == "A"


def test_author_filter(fragment_tree):
  root, write = fragment_tree
  write("a.md", "---\ntitle: A\ntype: todo\nauthor: user\n---\n")
  write("b.md", "---\ntitle: B\ntype: todo\nauthor: agent\n---\n")

  result = list_fragments.scan_fragments(str(root), author_filter="user")

  with_title = [f for f in result if "title" in f]
  assert len(with_title) == 1
  assert with_title[0]["title"] == "A"


def test_combined_filters(fragment_tree):
  root, write = fragment_tree
  write("a.md", "---\ntitle: A\ntype: todo\nstatus: active\n---\n")
  write("b.md", "---\ntitle: B\ntype: episode\nstatus: active\n---\n")
  write("c.md", "---\ntitle: C\ntype: todo\nstatus: blocked\n---\n")

  result = list_fragments.scan_fragments(
      str(root), type_filter="todo", status_filter="active"
  )

  with_title = [f for f in result if "title" in f]
  assert len(with_title) == 1
  assert with_title[0]["title"] == "A"


def test_adds_relative_path(fragment_tree):
  root, write = fragment_tree
  write("sub/c.md", "---\ntitle: C\n---\n")

  result = list_fragments.scan_fragments(str(root))

  assert result[0]["_path"] == os.path.join("sub", "c.md")


def test_empty_directory(fragment_tree):
  root, _ = fragment_tree
  assert list_fragments.scan_fragments(str(root)) == []


def test_max_depth_zero_lists_root_only(fragment_tree):
  root, write = fragment_tree
  write("a.md", "---\ntitle: A\n---\n")
  write("sub/b.md", "---\ntitle: B\n---\n")

  result = list_fragments.scan_fragments(str(root), max_depth=0)

  paths = [f["_path"] for f in result]
  assert "a.md" in paths
  # Subdirectory listed but not recursed into.
  assert any(f["_status"] == "directory" for f in result)
  assert not any(f.get("title") == "B" for f in result)


def test_max_depth_one_recurses_one_level(fragment_tree):
  root, write = fragment_tree
  write("a.md", "---\ntitle: A\n---\n")
  write("sub/b.md", "---\ntitle: B\n---\n")
  write("sub/deep/c.md", "---\ntitle: C\n---\n")

  result = list_fragments.scan_fragments(str(root), max_depth=1)

  titles = {f["title"] for f in result if "title" in f}
  assert titles == {"A", "B"}
  # sub/deep/ listed as a directory entry.
  dirs = [f for f in result if f.get("_status") == "directory"]
  assert len(dirs) == 1
  assert dirs[0]["_path"].startswith(os.path.join("sub", "deep"))


def test_max_depth_none_is_unlimited(fragment_tree):
  root, write = fragment_tree
  write("a/b/c/d.md", "---\ntitle: Deep\n---\n")

  result = list_fragments.scan_fragments(str(root), max_depth=None)

  assert any(f.get("title") == "Deep" for f in result)


# ---------------------------------------------------------------------------
# _parse_recurse
# ---------------------------------------------------------------------------


def test_parse_recurse_star():
  assert list_fragments._parse_recurse("*") is None


def test_parse_recurse_integer():
  assert list_fragments._parse_recurse("0") == 0
  assert list_fragments._parse_recurse("3") == 3


def test_parse_recurse_invalid():
  with pytest.raises(argparse.ArgumentTypeError):
    list_fragments._parse_recurse("abc")


def test_parse_recurse_negative():
  with pytest.raises(argparse.ArgumentTypeError):
    list_fragments._parse_recurse("-1")


# ---------------------------------------------------------------------------
# format_blocks
# ---------------------------------------------------------------------------


def test_format_blocks_empty():
  assert list_fragments.format_blocks([]) == "(no fragments found)"


def test_format_blocks_single():
  output = list_fragments.format_blocks([
      {"_path": "todos/a.md", "title": "Fix bug", "type": "todo"},
  ])
  assert "── todos/a.md" in output
  assert "   title: Fix bug" in output
  assert "   type: todo" in output
  # _path should not appear as a key: value line.
  assert "_path:" not in output


def test_format_blocks_multiple_separated_by_blank_line():
  output = list_fragments.format_blocks([
      {"_path": "a.md", "title": "A"},
      {"_path": "b.md", "title": "B"},
  ])
  assert "\n\n" in output
