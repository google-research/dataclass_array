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
"""Skill Debugger — traces skill usage through agent trajectories.

Given a trajectory (from file or API) and a skills directory, analyzes which
skills the agent loaded, read, followed, and which it should have used but
didn't. Produces a detailed report with effectiveness scores.

Usage:
  python skill_debugger.py --trajectory_file=steps.json
  python skill_debugger.py --trajectory_file=steps.json
  --skills_dir=/path/to/skills
  python skill_debugger.py --self_analyze --skills_dir=/path/to/skills
  python skill_debugger.py --trajectory_file=steps.json --json
"""

import argparse
import json
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skill_parser import parse_skill_directory
from trajectory_analyzer import (
    TrajectoryData,
    analyze_from_file,
    get_action_summary,
    get_files_accessed,
    get_skill_reads,
)
from skill_matcher import match_skills, summarize_matches


def _bar(value: float, width: int = 20) -> str:
  filled = int(value * width)
  return '█' * filled + '░' * (width - filled)


def _format_report(
    trajectory, results, summary, action_summary, files, skill_reads
):
  lines = []

  lines.append('╔' + '═' * 70 + '╗')
  lines.append('║' + '  SKILL DEBUGGER REPORT'.center(70) + '║')
  lines.append('╚' + '═' * 70 + '╝')
  lines.append('')

  lines.append(
      '┌─ TRAJECTORY OVERVIEW ─────────────────────────────────────────────┐'
  )
  lines.append(f'│  ID:        {trajectory.trajectory_id or "(local)"}')
  lines.append(f'│  Source:    {trajectory.source}')
  lines.append(f'│  Steps:     {trajectory.num_steps}')
  lines.append(f'│  Actions:   {sum(action_summary.values())}')
  lines.append('│')
  lines.append('│  Action Breakdown:')
  for action_type, count in sorted(action_summary.items(), key=lambda x: -x[1]):
    pct = count / max(1, sum(action_summary.values())) * 100
    lines.append(f'│    {action_type:<25} {count:>4}  ({pct:>5.1f}%)')
  lines.append(
      '└──────────────────────────────────────────────────────────────────┘'
  )
  lines.append('')

  lines.append(
      '┌─ SKILL UTILIZATION SUMMARY ───────────────────────────────────────┐'
  )
  lines.append(f'│  Total skills analyzed:      {summary["total_skills"]:>4}')
  lines.append(f'│  Skills read by agent:       {summary["skills_read"]:>4}')
  lines.append(f'│  Skills actively used:       {summary["skills_used"]:>4}')
  lines.append(
      '│  Relevant but ignored:      '
      f' {summary["skills_relevant_but_ignored"]:>4}'
  )
  lines.append(
      f'│  Average effectiveness:      {summary["average_effectiveness"]:.2f}'
  )
  lines.append('│')
  used = summary['skills_read']
  total = summary['total_skills']
  utilization = used / max(1, total)
  lines.append(f'│  Utilization:  {_bar(utilization)} {utilization*100:.0f}%')
  lines.append(
      '└──────────────────────────────────────────────────────────────────┘'
  )
  lines.append('')

  actively_used = [r for r in results if r.status == 'actively_used']
  if actively_used:
    lines.append(
        '┌─ ACTIVELY USED SKILLS ────────────────────────────────────────────┐'
    )
    for r in actively_used:
      lines.append(f'│')
      lines.append(f'│  🟢 {r.skill_name}')
      lines.append(
          f'│     Effectiveness: {_bar(r.effectiveness)} {r.effectiveness:.2f}'
      )
      lines.append(f'│     Read at step:  {r.read_at_step}')
      if r.commands_used:
        lines.append(f'│     Commands used: {", ".join(r.commands_used)}')
      if r.paths_accessed:
        lines.append(f'│     Paths touched: {len(r.paths_accessed)}')
      followed = [m for m in r.instructions_followed if m.followed]
      ignored = [m for m in r.instructions_followed if not m.followed]
      if followed:
        lines.append(
            '│     Instructions followed:'
            f' {len(followed)}/{len(r.instructions_followed)}'
        )
        for m in followed[:3]:
          wrapped = textwrap.shorten(m.instruction, 55)
          lines.append(f'│       ✓ {wrapped}')
      if ignored:
        lines.append(f'│     Instructions missed:')
        for m in ignored[:3]:
          wrapped = textwrap.shorten(m.instruction, 55)
          lines.append(f'│       ✗ {wrapped}')
    lines.append(
        '└──────────────────────────────────────────────────────────────────┘'
    )
    lines.append('')

  implicitly_used = [r for r in results if r.status == 'implicitly_used']
  if implicitly_used:
    lines.append(
        '┌─ IMPLICITLY USED (no explicit read, but artifacts used) ────────┐'
    )
    for r in implicitly_used:
      lines.append(f'│  🟡 {r.skill_name}')
      if r.commands_used:
        lines.append(f'│     Commands: {", ".join(r.commands_used[:3])}')
      if r.paths_accessed:
        lines.append(f'│     Paths: {", ".join(r.paths_accessed[:3])}')
    lines.append(
        '└──────────────────────────────────────────────────────────────────┘'
    )
    lines.append('')

  read_unused = [r for r in results if r.status == 'read_but_unused']
  if read_unused:
    lines.append(
        '┌─ READ BUT NOT USED ─────────────────────────────────────────────┐'
    )
    for r in read_unused:
      lines.append(f'│  🔵 {r.skill_name}  (read at step {r.read_at_step})')
    lines.append(
        '└──────────────────────────────────────────────────────────────────┘'
    )
    lines.append('')

  relevant_ignored = [r for r in results if r.status == 'relevant_but_ignored']
  if relevant_ignored:
    lines.append(
        '┌─ RELEVANT BUT IGNORED (potential gaps) ─────────────────────────┐'
    )
    for r in relevant_ignored:
      lines.append(
          f'│  🔴 {r.skill_name}  (relevance: {r.relevance_score:.2f})'
      )
    lines.append(
        '└──────────────────────────────────────────────────────────────────┘'
    )
    lines.append('')

  if skill_reads:
    lines.append(
        '┌─ SKILL FILE READS (chronological) ─────────────────────────────┐'
    )
    for path in skill_reads:
      short = path.split('/skills/')[-1] if '/skills/' in path else path
      lines.append(f'│  📖 {short}')
    lines.append(
        '└──────────────────────────────────────────────────────────────────┘'
    )
    lines.append('')

  lines.append(
      '┌─ RECOMMENDATIONS ────────────────────────────────────────────────┐'
  )
  has_recs = False
  if relevant_ignored:
    has_recs = True
    lines.append('│  ⚠ Skills relevant but not read:')
    for r in relevant_ignored[:5]:
      lines.append(
          f'│    → {r.skill_name} (relevance: {r.relevance_score:.2f})'
      )
  if read_unused:
    has_recs = True
    lines.append('│  ℹ Skills read but not applied:')
    for r in read_unused[:5]:
      lines.append(f'│    → {r.skill_name}')
  missed = []
  for r in actively_used:
    for m in r.instructions_followed:
      if not m.followed:
        missed.append((r.skill_name, m.instruction))
  if missed:
    has_recs = True
    lines.append('│  📝 Instructions not followed:')
    for sn, inst in missed[:5]:
      lines.append(f'│    → [{sn}] {textwrap.shorten(inst, 50)}')
  if not has_recs:
    lines.append('│  ✅ All relevant skills were read and followed!')
  lines.append(
      '└──────────────────────────────────────────────────────────────────┘'
  )

  return '\n'.join(lines)


def _self_analyze(skills_dir):
  skills = parse_skill_directory(skills_dir)
  lines = []
  lines.append('╔' + '═' * 70 + '╗')
  lines.append('║' + '  SKILL SYSTEM SELF-ANALYSIS'.center(70) + '║')
  lines.append('╚' + '═' * 70 + '╝')
  lines.append(f'\nAnalyzed {len(skills)} skills from {skills_dir}\n')

  lines.append(
      '┌─ SKILL SIZES ─────────────────────────────────────────────────────┐'
  )
  for s in sorted(skills, key=lambda s: -s.content_lines)[:15]:
    bar = _bar(min(1.0, s.content_lines / 300), 15)
    lines.append(f'│  {s.name:<30} {s.content_lines:>4} lines  {bar}')
  lines.append(
      '└──────────────────────────────────────────────────────────────────┘\n'
  )

  lines.append(
      '┌─ TRIGGER COVERAGE ────────────────────────────────────────────────┐'
  )
  with_t = [s for s in skills if s.triggers]
  without_t = [s for s in skills if not s.triggers]
  lines.append(f'│  With triggers:    {len(with_t):>4}')
  lines.append(f'│  Without triggers: {len(without_t):>4}')
  if without_t:
    lines.append('│  Missing:')
    for s in without_t[:15]:
      lines.append(f'│    ⚠ {s.name}')
  lines.append(
      '└──────────────────────────────────────────────────────────────────┘\n'
  )

  lines.append(
      '┌─ CLI TOOLS ───────────────────────────────────────────────────────┐'
  )
  with_cli = [s for s in skills if s.cli_commands]
  lines.append(f'│  Skills with CLI commands: {len(with_cli)}')
  for s in sorted(with_cli, key=lambda x: -len(x.cli_commands))[:10]:
    cmds = ', '.join(c.binary_path[:40] for c in s.cli_commands[:3])
    lines.append(f'│    {s.name:<25} → {cmds}')
  lines.append(
      '└──────────────────────────────────────────────────────────────────┘\n'
  )

  lines.append(
      '┌─ INSTRUCTION DENSITY ─────────────────────────────────────────────┐'
  )
  by_inst = sorted(skills, key=lambda s: -len(s.key_instructions))
  for s in by_inst[:10]:
    lines.append(f'│  {s.name:<30} {len(s.key_instructions):>3} instructions')
  lines.append(
      '└──────────────────────────────────────────────────────────────────┘\n'
  )

  total_triggers = sum(len(s.triggers) for s in skills)
  total_cmds = sum(len(s.cli_commands) for s in skills)
  total_instructions = sum(len(s.key_instructions) for s in skills)
  total_paths = sum(len(s.referenced_paths) for s in skills)
  total_xrefs = sum(len(s.cross_references) for s in skills)
  lines.append(
      '┌─ TOTALS ──────────────────────────────────────────────────────────┐'
  )
  lines.append(f'│  Skills:         {len(skills):>5}')
  lines.append(f'│  Triggers:       {total_triggers:>5}')
  lines.append(f'│  CLI commands:   {total_cmds:>5}')
  lines.append(f'│  Instructions:   {total_instructions:>5}')
  lines.append(f'│  Path refs:      {total_paths:>5}')
  lines.append(f'│  Cross-refs:     {total_xrefs:>5}')
  lines.append(
      '└──────────────────────────────────────────────────────────────────┘'
  )

  return '\n'.join(lines)


def main():
  parser = argparse.ArgumentParser(description='Skill Debugger')
  parser.add_argument('--trajectory_file', help='Path to trajectory JSON')
  parser.add_argument('--skills_dir', help='Path to skills directory')
  parser.add_argument('--json', action='store_true', help='JSON output')
  parser.add_argument(
      '--self_analyze', action='store_true', help='Self-analysis mode'
  )
  args = parser.parse_args()

  default_dirs = [
      os.path.expanduser('~/_agents/skills'),
      os.path.join(
          os.path.dirname(os.path.abspath(__file__)), '..', '..', 'skills'
      ),
  ]

  if args.self_analyze:
    skills_dir = args.skills_dir
    if not skills_dir:
      for d in default_dirs:
        if os.path.isdir(d):
          skills_dir = d
          break
    if not skills_dir:
      print('ERROR: --skills_dir required', file=sys.stderr)
      sys.exit(1)
    print(_self_analyze(skills_dir))
    return

  if not args.trajectory_file:
    print(
        'ERROR: --trajectory_file or --self_analyze required', file=sys.stderr
    )
    sys.exit(1)

  trajectory = analyze_from_file(args.trajectory_file)
  action_summary = get_action_summary(trajectory)
  files = get_files_accessed(trajectory)
  skill_reads = get_skill_reads(trajectory)

  skills_dir = args.skills_dir
  if not skills_dir:
    for d in default_dirs:
      if os.path.isdir(d):
        skills_dir = d
        break
  if not skills_dir:
    print('ERROR: No skills directory found', file=sys.stderr)
    sys.exit(1)

  skills = parse_skill_directory(skills_dir)
  results = match_skills(skills, trajectory)
  summary = summarize_matches(results)

  if args.json:
    output = {
        'trajectory_id': trajectory.trajectory_id,
        'num_steps': trajectory.num_steps,
        'action_summary': action_summary,
        'skill_summary': summary,
        'skill_results': [
            {
                'name': r.skill_name,
                'status': r.status,
                'was_read': r.was_read,
                'effectiveness': r.effectiveness,
                'relevance': r.relevance_score,
                'commands_used': r.commands_used,
            }
            for r in results
            if r.status != 'unused'
        ],
    }
    print(json.dumps(output, indent=2))
  else:
    print(
        _format_report(
            trajectory, results, summary, action_summary, files, skill_reads
        )
    )


if __name__ == '__main__':
  main()
