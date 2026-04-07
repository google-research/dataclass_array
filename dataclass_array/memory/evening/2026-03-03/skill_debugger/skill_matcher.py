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

"""Matches agent trajectory actions against parsed skill definitions.

Determines:
- Which skills were actively used (agent read the skill AND followed its
instructions)
- Which skills were relevant but ignored (skill would have helped but agent
didn't use it)
- Which skill instructions were followed vs. ignored
- Effectiveness score per skill
"""

from dataclasses import dataclass, field
import re

from skill_parser import CLICommand, ParsedSkill
from trajectory_analyzer import ActionType, AgentAction, TrajectoryData


@dataclass
class InstructionMatch:
  instruction: str
  followed: bool
  evidence: str = ''


@dataclass
class SkillMatchResult:
  skill_name: str
  was_read: bool
  read_at_step: int | None = None
  relevance_score: float = 0.0
  instructions_followed: list[InstructionMatch] = field(default_factory=list)
  commands_used: list[str] = field(default_factory=list)
  paths_accessed: list[str] = field(default_factory=list)
  effectiveness: float = 0.0
  status: str = 'unused'


def _normalize(text: str) -> str:
  return re.sub(r'\s+', ' ', text.lower().strip())


def _text_overlap(text_a: str, text_b: str, min_word_len: int = 4) -> float:
  words_a = {w for w in _normalize(text_a).split() if len(w) >= min_word_len}
  words_b = {w for w in _normalize(text_b).split() if len(w) >= min_word_len}
  if not words_a or not words_b:
    return 0.0
  overlap = words_a & words_b
  return len(overlap) / min(len(words_a), len(words_b))


def _was_skill_read(
    skill: ParsedSkill,
    actions: list[AgentAction],
) -> tuple[bool, int | None]:
  skill_dir_name = skill.name
  for action in actions:
    if action.action_type != ActionType.SKILL_READ:
      continue
    path = action.file_path.lower()
    if (
        f'/skills/{skill_dir_name}/' in path
        or f'/{skill_dir_name}/skill.md' in path
    ):
      return True, action.step_index
  return False, None


def _check_command_usage(
    cli_commands: list[CLICommand],
    actions: list[AgentAction],
) -> list[str]:
  used = []
  command_actions = [
      a for a in actions if a.action_type == ActionType.COMMAND_RUN
  ]
  for cmd in cli_commands:
    binary = cmd.binary_path.lower()
    for action in command_actions:
      content = action.content.lower()
      if binary in content or any(f in content for f in cmd.flags[:3]):
        used.append(cmd.binary_path)
        break
  return used


def _check_path_access(
    referenced_paths: list[str],
    actions: list[AgentAction],
) -> list[str]:
  accessed = []
  file_actions = [a for a in actions if a.file_path]
  for ref_path in referenced_paths:
    ref_parts = ref_path.lower().rstrip('/').split('/')
    if len(ref_parts) < 2:
      continue
    ref_suffix = '/'.join(ref_parts[-2:])
    for action in file_actions:
      if ref_suffix in action.file_path.lower():
        accessed.append(ref_path)
        break
  return accessed


def _compute_relevance(
    skill: ParsedSkill,
    user_request: str,
) -> float:
  if not user_request:
    return 0.0
  trigger_score = 0.0
  for trigger in skill.triggers:
    if trigger in user_request.lower():
      trigger_score = max(trigger_score, 0.8)
    elif _text_overlap(trigger, user_request) > 0.3:
      trigger_score = max(trigger_score, 0.4)
  desc_score = _text_overlap(skill.description, user_request)
  return min(1.0, trigger_score + desc_score * 0.5)


def _check_instructions(
    skill: ParsedSkill,
    actions: list[AgentAction],
) -> list[InstructionMatch]:
  matches = []
  all_content = ' '.join(
      a.content
      for a in actions
      if a.content
      and a.action_type
      in (
          ActionType.MODEL_RESPONSE,
          ActionType.COMMAND_RUN,
          ActionType.FILE_WRITE,
          ActionType.FILE_EDIT,
      )
  ).lower()
  all_files = ' '.join(a.file_path for a in actions if a.file_path).lower()
  all_tool_names = ' '.join(a.tool_name for a in actions if a.tool_name).lower()
  combined = f'{all_content} {all_files} {all_tool_names}'

  for instruction in skill.key_instructions[:20]:
    key_terms = [w for w in _normalize(instruction).split() if len(w) >= 4]
    if not key_terms:
      continue
    match_count = sum(1 for t in key_terms if t in combined)
    threshold = max(1, len(key_terms) // 3)
    followed = match_count >= threshold
    matches.append(
        InstructionMatch(
            instruction=instruction[:120],
            followed=followed,
            evidence=f'{match_count}/{len(key_terms)} key terms found'
            if followed
            else '',
        )
    )
  return matches


def match_skills(
    skills: list[ParsedSkill],
    trajectory: TrajectoryData,
) -> list[SkillMatchResult]:
  user_requests = ' '.join(
      a.content
      for a in trajectory.actions
      if a.action_type == ActionType.USER_INPUT
  )

  results = []
  for skill in skills:
    was_read, read_step = _was_skill_read(skill, trajectory.actions)
    relevance = _compute_relevance(skill, user_requests)
    commands_used = _check_command_usage(skill.cli_commands, trajectory.actions)
    paths_accessed = _check_path_access(
        skill.referenced_paths, trajectory.actions
    )
    instruction_matches = _check_instructions(skill, trajectory.actions)

    followed_count = sum(1 for m in instruction_matches if m.followed)
    total_instructions = len(instruction_matches)
    instruction_rate = (
        followed_count / total_instructions if total_instructions > 0 else 0.0
    )

    if was_read:
      effectiveness = (
          0.3 * min(1.0, len(commands_used) / max(1, len(skill.cli_commands)))
          + 0.3 * instruction_rate
          + 0.2
          * min(1.0, len(paths_accessed) / max(1, len(skill.referenced_paths)))
          + 0.2 * relevance
      )
      status = 'actively_used' if effectiveness > 0.2 else 'read_but_unused'
    elif relevance > 0.3:
      effectiveness = 0.0
      status = 'relevant_but_ignored'
    elif commands_used or paths_accessed:
      effectiveness = 0.3
      status = 'implicitly_used'
    else:
      effectiveness = 0.0
      status = 'unused'

    results.append(
        SkillMatchResult(
            skill_name=skill.name,
            was_read=was_read,
            read_at_step=read_step,
            relevance_score=round(relevance, 2),
            instructions_followed=instruction_matches,
            commands_used=commands_used,
            paths_accessed=paths_accessed,
            effectiveness=round(effectiveness, 2),
            status=status,
        )
    )

  results.sort(key=lambda r: (-r.effectiveness, -r.relevance_score))
  return results


def summarize_matches(results: list[SkillMatchResult]) -> dict:
  by_status = {}
  for r in results:
    by_status.setdefault(r.status, []).append(r.skill_name)
  total_read = sum(1 for r in results if r.was_read)
  total_used = sum(
      1 for r in results if r.status in ('actively_used', 'implicitly_used')
  )
  total_relevant_ignored = sum(
      1 for r in results if r.status == 'relevant_but_ignored'
  )
  avg_effectiveness = sum(r.effectiveness for r in results if r.was_read) / max(
      1, total_read
  )
  return {
      'total_skills': len(results),
      'skills_read': total_read,
      'skills_used': total_used,
      'skills_relevant_but_ignored': total_relevant_ignored,
      'average_effectiveness': round(avg_effectiveness, 2),
      'by_status': by_status,
  }
