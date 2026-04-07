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
"""Skill Quality Scorer — rates every skill and suggests improvements.

Dimensions:
  1. Discoverability — triggers, description clarity
  2. Actionability — CLI commands, instructions, examples
  3. Connectivity — cross-references to/from other skills
  4. Completeness — frontmatter, headers, usage, examples
  5. Density — information per line

Usage:
  python3 skill_quality_scorer.py --skills_dir=/path/to/skills
  python3 skill_quality_scorer.py --skills_dir=/path/to/skills --fix
  python3 skill_quality_scorer.py --skills_dir=/path/to/skills --json
"""

import argparse
from dataclasses import dataclass, field
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from skill_parser import ParsedSkill, parse_skill_directory


@dataclass
class Dimension:
  name: str
  score: float
  max_score: float
  issues: list[str] = field(default_factory=list)
  fixes: list[str] = field(default_factory=list)


@dataclass
class SkillQuality:
  skill_name: str
  path: str
  overall_score: float = 0.0
  grade: str = ''
  dimensions: list[Dimension] = field(default_factory=list)
  top_issues: list[str] = field(default_factory=list)
  top_fixes: list[str] = field(default_factory=list)


def _read_content(path: str) -> str:
  if path and os.path.isfile(path):
    with open(path) as f:
      return f.read()
  return ''


def _score_discoverability(skill: ParsedSkill) -> Dimension:
  s, issues, fixes = 0.0, [], []
  if skill.triggers:
    s += 1.0
    if len(skill.triggers) >= 3:
      s += 0.5
  else:
    issues.append('No trigger phrases')
    fixes.append('Add "Use when ..." or "Triggers on: ..." to description')
  if skill.description and len(skill.description) >= 20:
    s += 1.0
  else:
    issues.append('Description missing or too short')
    fixes.append('Add a clear description in YAML frontmatter')
  if any(
      "don't use" in t.lower() or 'do not use' in t.lower()
      for t in skill.triggers
  ):
    s += 0.5
  elif skill.triggers:
    issues.append('No negative triggers')
    fixes.append('Add "Don\'t use for ..." to reduce false activations')
  return Dimension('discoverability', s, 3.0, issues, fixes)


def _score_actionability(skill: ParsedSkill) -> Dimension:
  s, issues, fixes = 0.0, [], []
  if skill.cli_commands:
    s += 1.5
    if any(c.flags for c in skill.cli_commands):
      s += 0.5
  if skill.key_instructions:
    s += 1.0
    if len(skill.key_instructions) >= 3:
      s += 0.5
  else:
    issues.append('No imperative instructions')
    fixes.append('Add MUST/ALWAYS/NEVER instructions')
  content = _read_content(skill.path)
  if '```' in content:
    s += 0.5
  else:
    issues.append('No code examples')
    fixes.append('Add a fenced code block with a concrete example')
  return Dimension('actionability', s, 4.0, issues, fixes)


def _score_connectivity(skill, all_names, incoming) -> Dimension:
  s, issues, fixes = 0.0, [], []
  valid_out = [r for r in skill.cross_references if r in all_names]
  if valid_out:
    s += 1.0
    if len(valid_out) >= 2:
      s += 0.5
  else:
    issues.append('No outgoing cross-references')
    fixes.append('Reference related skills with backtick notation')
  if incoming.get(skill.name):
    s += 1.0
  else:
    issues.append('No incoming references from other skills')
    fixes.append(f'Have related skills reference `{skill.name}`')
  broken = [
      r for r in skill.cross_references if r not in all_names and len(r) > 3
  ]
  if not broken:
    s += 0.5
  else:
    issues.append(f'{len(broken)} broken refs')
    fixes.append(f'Fix: {", ".join(broken[:3])}')
  return Dimension('connectivity', s, 3.0, issues, fixes)


def _score_completeness(skill: ParsedSkill) -> Dimension:
  s, issues, fixes = 0.0, [], []
  content = _read_content(skill.path)
  if not content:
    return Dimension('completeness', 0, 3.0, ['File not found'], [])
  if content.startswith('---'):
    s += 0.5
  else:
    issues.append('No YAML frontmatter')
    fixes.append('Add --- frontmatter with name/description')
  if re.search(r'^##\s', content, re.MULTILINE):
    s += 0.5
  else:
    issues.append('No section headers')
    fixes.append('Add ## headers')
  if re.search(r'(?i)(usage|quick start|how to)', content):
    s += 1.0
  else:
    issues.append('No usage section')
    fixes.append('Add Quick Start or Usage section')
  if re.search(r'(?i)(example|sample|demo)', content):
    s += 1.0
  else:
    issues.append('No examples')
    fixes.append('Add concrete examples')
  return Dimension('completeness', s, 3.0, issues, fixes)


def _score_density(skill: ParsedSkill) -> Dimension:
  s, issues, fixes = 0.0, [], []
  lines = skill.content_lines
  if lines == 0:
    return Dimension('density', 0, 2.0, ['Empty'], ['Add content'])
  info = (
      len(skill.triggers)
      + len(skill.key_instructions)
      + len(skill.cli_commands)
      + len(skill.cross_references)
  )
  density = info / lines
  if 20 <= lines <= 300:
    s += 1.0
  elif lines < 20:
    issues.append(f'Very short ({lines} lines)')
    fixes.append('Expand with detail and examples')
  else:
    issues.append(f'Very long ({lines} lines)')
    fixes.append('Consider splitting or trimming')
  if density >= 0.05:
    s += 1.0
  elif density >= 0.02:
    s += 0.5
    issues.append('Low info density')
    fixes.append('Add more actionable content per line')
  else:
    issues.append('Very low info density')
    fixes.append('Content is mostly prose without actionable info')
  return Dimension('density', s, 2.0, issues, fixes)


def _grade(score, max_score):
  pct = score / max_score if max_score > 0 else 0
  if pct >= 0.85:
    return 'A'
  if pct >= 0.7:
    return 'B'
  if pct >= 0.55:
    return 'C'
  if pct >= 0.4:
    return 'D'
  return 'F'


def _bar(val, w=15):
  f = int(val * w)
  return '█' * f + '░' * (w - f)


def score_skills(skills):
  all_names = {s.name for s in skills}
  incoming = {}
  for s in skills:
    for ref in s.cross_references:
      if ref in all_names:
        incoming.setdefault(ref, []).append(s.name)
  results = []
  for skill in skills:
    dims = [
        _score_discoverability(skill),
        _score_actionability(skill),
        _score_connectivity(skill, all_names, incoming),
        _score_completeness(skill),
        _score_density(skill),
    ]
    total = sum(d.score for d in dims)
    mx = sum(d.max_score for d in dims)
    all_issues = [i for d in dims for i in d.issues]
    all_fixes = [f for d in dims for f in d.fixes]
    results.append(
        SkillQuality(
            skill_name=skill.name,
            path=skill.path,
            overall_score=round(total, 1),
            grade=_grade(total, mx),
            dimensions=dims,
            top_issues=all_issues[:5],
            top_fixes=all_fixes[:5],
        )
    )
  results.sort(key=lambda r: -r.overall_score)
  return results


def format_report(results, show_fixes=False):
  lines = []
  mx = 15.0
  lines.append('╔' + '═' * 70 + '╗')
  lines.append('║' + '  SKILL QUALITY REPORT'.center(70) + '║')
  lines.append('╚' + '═' * 70 + '╝\n')

  gc = {}
  for r in results:
    gc[r.grade] = gc.get(r.grade, 0) + 1
  avg = sum(r.overall_score for r in results) / max(1, len(results))

  lines.append(
      '┌─ OVERVIEW ────────────────────────────────────────────────────────┐'
  )
  lines.append(
      f'│  Skills: {len(results)}   Average: {avg:.1f}/{mx:.0f}'
      f' ({avg/mx*100:.0f}%)'
  )
  for g in ['A', 'B', 'C', 'D', 'F']:
    c = gc.get(g, 0)
    if c:
      lines.append(f'│    {g}: {c:>3} {"█" * c}')
  lines.append(
      '└──────────────────────────────────────────────────────────────────┘\n'
  )

  lines.append(
      '┌─ RANKINGS ────────────────────────────────────────────────────────┐'
  )
  for r in results:
    lines.append(
        f'│  {r.grade} {r.overall_score:>4.1f}/{mx:.0f} '
        f' {_bar(r.overall_score/mx)}  {r.skill_name}'
    )
  lines.append(
      '└──────────────────────────────────────────────────────────────────┘\n'
  )

  bottom = [r for r in results if r.grade in ('D', 'F')]
  if bottom:
    lines.append(
        '┌─ NEEDS IMPROVEMENT ─────────────────────────────────────────────┐'
    )
    for r in bottom:
      lines.append(f'│  🔴 {r.skill_name} ({r.grade}, {r.overall_score})')
      for d in r.dimensions:
        if d.issues:
          lines.append(f'│    {d.name}: {d.score:.0f}/{d.max_score:.0f}')
          for i in d.issues[:2]:
            lines.append(f'│      ⚠ {i}')
      if show_fixes and r.top_fixes:
        for f in r.top_fixes[:3]:
          lines.append(f'│      → {f[:65]}')
    lines.append(
        '└──────────────────────────────────────────────────────────────────┘\n'
    )

  dim_names = [
      'discoverability',
      'actionability',
      'connectivity',
      'completeness',
      'density',
  ]
  dim_avgs = {}
  for dn in dim_names:
    vals = [
        d.score / d.max_score
        for r in results
        for d in r.dimensions
        if d.name == dn and d.max_score > 0
    ]
    dim_avgs[dn] = sum(vals) / max(1, len(vals))
  lines.append(
      '┌─ DIMENSION AVERAGES ──────────────────────────────────────────────┐'
  )
  for dn in dim_names:
    lines.append(f'│  {dn:<20} {_bar(dim_avgs[dn])} {dim_avgs[dn]*100:.0f}%')
  lines.append(
      '└──────────────────────────────────────────────────────────────────┘\n'
  )

  worst = min(dim_avgs, key=dim_avgs.get)
  nt = sum(
      1
      for r in results
      for d in r.dimensions
      if d.name == 'discoverability' and 'No trigger phrases' in d.issues
  )
  ne = sum(
      1
      for r in results
      for d in r.dimensions
      if d.name == 'completeness' and 'No examples' in d.issues
  )
  no = sum(
      1
      for r in results
      for d in r.dimensions
      if d.name == 'connectivity' and 'No outgoing cross-references' in d.issues
  )
  lines.append(
      '┌─ RECOMMENDATIONS ────────────────────────────────────────────────┐'
  )
  lines.append(f'│  Weakest: {worst} ({dim_avgs[worst]*100:.0f}%)')
  if nt:
    lines.append(f'│  → {nt} skills need trigger phrases')
  if ne:
    lines.append(f'│  → {ne} skills need examples')
  if no:
    lines.append(f'│  → {no} skills need cross-references')
  lines.append(
      '└──────────────────────────────────────────────────────────────────┘'
  )
  return '\n'.join(lines)


def main():
  p = argparse.ArgumentParser(description='Skill Quality Scorer')
  p.add_argument('--skills_dir', required=True)
  p.add_argument('--json', action='store_true')
  p.add_argument('--fix', action='store_true')
  args = p.parse_args()
  skills = parse_skill_directory(args.skills_dir)
  results = score_skills(skills)
  if args.json:
    print(
        json.dumps(
            [
                {
                    'name': r.skill_name,
                    'grade': r.grade,
                    'score': r.overall_score,
                    'dims': {
                        d.name: {
                            'score': d.score,
                            'max': d.max_score,
                            'issues': d.issues,
                        }
                        for d in r.dimensions
                    },
                    'fixes': r.top_fixes,
                }
                for r in results
            ],
            indent=2,
        )
    )
  else:
    print(format_report(results, show_fixes=args.fix))


if __name__ == '__main__':
  main()
