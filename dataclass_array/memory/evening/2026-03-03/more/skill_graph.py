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

#!/usr/bin/env python3
"""Skill Graph Analyzer — Map, analyze, and debug the agent skill system.

The update-skills SKILL.md describes skills as a program graph:
- Skills are nodes
- Cross-references are edges
- AGENTS.md is main()

But there's no tooling to actually see or analyze this graph. This tool
builds the graph from source and finds structural problems:
- Orphaned skills (unreachable from AGENTS.md)
- Missing edges (skills that should cross-reference each other but don't)
- Potential duplicate coverage
- Dead ends (skills that reference nonexistent targets)

Usage:
  python3 skill_graph.py [--agents-md PATH] [--skills-dir PATH ...]

  Defaults to scanning configs/users/epot/_agents/ for AGENTS.md and skills.
"""

import argparse
from collections import defaultdict
import os
import re
import sys


BACKTICK_REF = re.compile(r'`([a-z][a-z0-9_-]*(?:/[a-z][a-z0-9_-]*)*)`')
SKILL_HEADER = re.compile(r'^name:\s*(.+)\s*$', re.MULTILINE)
DESCRIPTION_HEADER = re.compile(
    r'^description:\s*[>|]?\s*\n((?:\s+.+\n)*)', re.MULTILINE
)
DESCRIPTION_INLINE = re.compile(r'^description:\s*(.+)\s*$', re.MULTILINE)


class SkillNode:

  def __init__(self, name, path, content=''):
    self.name = name
    self.path = path
    self.content = content
    self.references_out = set()
    self.references_in = set()
    self.keywords = set()
    self.line_count = len(content.splitlines()) if content else 0
    self.description = ''


class SkillGraph:

  def __init__(self):
    self.nodes = {}
    self.agents_md_refs = set()
    self.all_issues = []

  def add_node(self, name, path, content):
    node = SkillNode(name, path, content)
    self.nodes[name] = node

    m = DESCRIPTION_INLINE.search(content)
    if m:
      node.description = m.group(1).strip()
    else:
      m = DESCRIPTION_HEADER.search(content)
      if m:
        desc = m.group(1).strip()
        desc = re.sub(r'\s+', ' ', desc)
        node.description = desc

    refs = BACKTICK_REF.findall(content)
    for ref in refs:
      ref_clean = ref.strip().lower().replace('/', '-')
      if ref_clean != name and len(ref_clean) > 2:
        node.references_out.add(ref_clean)

    words = re.findall(r'\b[a-z]{4,}\b', content.lower())
    common_words = {
        'this',
        'that',
        'with',
        'from',
        'have',
        'been',
        'will',
        'your',
        'when',
        'what',
        'they',
        'into',
        'also',
        'then',
        'than',
        'each',
        'more',
        'some',
        'make',
        'like',
        'just',
        'only',
        'very',
        'well',
        'should',
        'would',
        'could',
        'before',
        'after',
        'about',
        'other',
        'these',
        'those',
        'there',
        'their',
        'where',
        'which',
        'while',
        'does',
        'done',
        'here',
        'most',
        'such',
        'over',
        'same',
        'even',
        'used',
        'using',
        'uses',
        'need',
        'file',
        'files',
        'path',
        'note',
        'must',
        'skill',
        'skills',
        'agent',
        'agents',
        'tool',
        'tools',
    }
    for w in words:
      if w not in common_words:
        node.keywords.add(w)

    return node

  def parse_agents_md(self, content):
    refs = BACKTICK_REF.findall(content)
    for ref in refs:
      ref_clean = ref.strip().lower().replace('/', '-')
      if len(ref_clean) > 2:
        self.agents_md_refs.add(ref_clean)

  def build_reverse_edges(self):
    for name, node in self.nodes.items():
      for ref in node.references_out:
        if ref in self.nodes:
          self.nodes[ref].references_in.add(name)

  def find_orphans(self):
    reachable = set()
    queue = list(self.agents_md_refs & set(self.nodes.keys()))
    reachable.update(queue)

    while queue:
      current = queue.pop(0)
      if current not in self.nodes:
        continue
      for ref in self.nodes[current].references_out:
        if ref in self.nodes and ref not in reachable:
          reachable.add(ref)
          queue.append(ref)

    orphans = set(self.nodes.keys()) - reachable
    return orphans

  def find_broken_refs(self):
    broken = []
    all_names = set(self.nodes.keys())
    for name, node in self.nodes.items():
      for ref in node.references_out:
        if ref not in all_names:
          close = [
              n
              for n in all_names
              if n != ref
              and (ref in n or n in ref or self._edit_distance_ok(ref, n))
          ]
          broken.append((name, ref, close))
    return broken

  def find_potential_missing_edges(self, threshold=0.3):
    missing = []
    names = list(self.nodes.keys())
    for i in range(len(names)):
      for j in range(i + 1, len(names)):
        a, b = self.nodes[names[i]], self.nodes[names[j]]
        if not a.keywords or not b.keywords:
          continue
        overlap = a.keywords & b.keywords
        min_size = min(len(a.keywords), len(b.keywords))
        if min_size == 0:
          continue
        similarity = len(overlap) / min_size
        if similarity > threshold:
          if (
              names[j] not in a.references_out
              and names[i] not in b.references_out
          ):
            top_shared = sorted(overlap)[:8]
            missing.append((names[i], names[j], similarity, top_shared))
    missing.sort(key=lambda x: -x[2])
    return missing[:15]

  def find_dead_ends(self):
    dead_ends = []
    for name, node in self.nodes.items():
      if not node.references_out and not node.references_in:
        dead_ends.append(name)
    return dead_ends

  def _edit_distance_ok(self, a, b):
    if abs(len(a) - len(b)) > 2:
      return False
    diffs = sum(1 for ca, cb in zip(a, b) if ca != cb)
    return diffs <= 2

  def analyze(self):
    self.build_reverse_edges()

    print()
    print('═══ SKILL GRAPH ANALYSIS ═══')
    print()
    print(f'Total skills: {len(self.nodes)}')
    print(f'AGENTS.md references: {len(self.agents_md_refs)}')
    total_lines = sum(n.line_count for n in self.nodes.values())
    print(f'Total content: {total_lines} lines')
    print()

    print('─── Skills by size ───')
    by_size = sorted(self.nodes.values(), key=lambda n: -n.line_count)
    for n in by_size[:10]:
      refs = len(n.references_out)
      print(f'  {n.name:30s} {n.line_count:4d} lines, {refs:2d} outgoing refs')
    print()

    print('─── Most referenced ───')
    by_refs = sorted(self.nodes.values(), key=lambda n: -len(n.references_in))
    for n in by_refs[:10]:
      incoming = len(n.references_in)
      if incoming > 0:
        sources = ', '.join(sorted(n.references_in)[:5])
        print(f'  {n.name:30s} ← {incoming} refs ({sources})')
    print()

    orphans = self.find_orphans()
    if orphans:
      print(f'─── Orphaned skills ({len(orphans)}) ───')
      print('  Not reachable from AGENTS.md:')
      for name in sorted(orphans):
        print(f'    • {name}')
      print()

    broken = self.find_broken_refs()
    if broken:
      print(f'─── Broken references ({len(broken)}) ───')
      for src, ref, close in broken:
        close_str = f' (did you mean: {", ".join(close)})' if close else ''
        print(f'  {src} → `{ref}`{close_str}')
      print()

    dead = self.find_dead_ends()
    if dead:
      print(f'─── Isolated skills ({len(dead)}) ───')
      print('  No incoming or outgoing references:')
      for name in sorted(dead):
        print(f'    • {name}')
      print()

    missing = self.find_potential_missing_edges()
    if missing:
      print(f'─── Potential missing edges ───')
      print('  Skills with high keyword overlap but no cross-reference:')
      for a, b, sim, keywords in missing[:10]:
        kw = ', '.join(keywords[:5])
        print(f'  {a} ↔ {b} ({sim:.0%} overlap: {kw})')
      print()

    agent_refs_not_found = self.agents_md_refs - set(self.nodes.keys())
    if agent_refs_not_found:
      print(f'─── AGENTS.md references not found as skills ───')
      for ref in sorted(agent_refs_not_found):
        print(f'    • `{ref}`')
      print()

    print('─── Graph Statistics ───')
    total_edges = sum(len(n.references_out) for n in self.nodes.values())
    avg_out = total_edges / max(len(self.nodes), 1)
    avg_in = sum(len(n.references_in) for n in self.nodes.values()) / max(
        len(self.nodes), 1
    )
    print(f'  Total edges: {total_edges}')
    print(f'  Average outgoing refs: {avg_out:.1f}')
    print(f'  Average incoming refs: {avg_in:.1f}')
    connectivity = 1 - (len(orphans) / max(len(self.nodes), 1))
    print(f'  Connectivity: {connectivity:.0%} reachable from AGENTS.md')
    print()

    print('═══════════════════════════')
    print()


def scan_directory(path):
  skills = []
  if not os.path.isdir(path):
    return skills

  for entry in os.listdir(path):
    skill_dir = os.path.join(path, entry)
    skill_file = os.path.join(skill_dir, 'SKILL.md')
    if os.path.isfile(skill_file):
      with open(skill_file, 'r') as f:
        content = f.read()
      m = SKILL_HEADER.search(content)
      name = m.group(1).strip() if m else entry
      skills.append((name, skill_file, content))
  return skills


def main():
  parser = argparse.ArgumentParser(description='Analyze the agent skill graph')
  parser.add_argument(
      '--agents-md',
      default=None,
      help='Path to AGENTS.md',
  )
  parser.add_argument(
      '--skills-dir',
      nargs='*',
      default=None,
      help='Directories containing skill subdirectories',
  )
  args = parser.parse_args()

  base = os.environ.get('AGENT_BASE', 'configs/users/epot/_agents')

  if args.agents_md:
    agents_md_path = args.agents_md
  else:
    candidates = [
        os.path.join(base, 'AGENTS.md'),
        'AGENTS.md',
    ]
    agents_md_path = None
    for c in candidates:
      if os.path.exists(c):
        agents_md_path = c
        break

  if args.skills_dir:
    skills_dirs = args.skills_dir
  else:
    skills_dirs = [os.path.join(base, 'skills')]

  graph = SkillGraph()

  if agents_md_path and os.path.exists(agents_md_path):
    with open(agents_md_path, 'r') as f:
      content = f.read()
    graph.parse_agents_md(content)
    print(f'Loaded AGENTS.md from {agents_md_path}')
  else:
    print('Warning: AGENTS.md not found')

  total_loaded = 0
  for skills_dir in skills_dirs:
    skills = scan_directory(skills_dir)
    for name, path, content in skills:
      graph.add_node(name, path, content)
      total_loaded += 1

  print(f'Loaded {total_loaded} skills from {len(skills_dirs)} directories')

  graph.analyze()


if __name__ == '__main__':
  main()
