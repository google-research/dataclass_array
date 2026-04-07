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

"""Analyzes trajectory data to extract the sequence of agent actions.

Works with two data sources:
1. Trajectory API (via trajectory_id) — fetches from the trajectory service
2. Local reflection API (via conversation_id) — reads from the running session
3. Raw JSON files — for offline analysis
"""

from dataclasses import dataclass, field
from enum import Enum
import json
import os
import re
import subprocess


class ActionType(Enum):
  USER_INPUT = 'user_input'
  TOOL_CALL = 'tool_call'
  FILE_VIEW = 'file_view'
  FILE_WRITE = 'file_write'
  FILE_EDIT = 'file_edit'
  COMMAND_RUN = 'command_run'
  COMMAND_STATUS = 'command_status'
  TASK_BOUNDARY = 'task_boundary'
  CODE_SEARCH = 'code_search'
  BUILD = 'build'
  TEST = 'test'
  NOTIFY_USER = 'notify_user'
  SKILL_READ = 'skill_read'
  MODEL_RESPONSE = 'model_response'
  OTHER = 'other'


@dataclass
class AgentAction:
  action_type: ActionType
  step_index: int
  timestamp: str = ''
  tool_name: str = ''
  arguments: dict = field(default_factory=dict)
  content: str = ''
  file_path: str = ''
  success: bool = True


@dataclass
class TrajectoryData:
  trajectory_id: str = ''
  source: str = ''
  timestamp: str = ''
  num_steps: int = 0
  num_executions: int = 0
  actions: list[AgentAction] = field(default_factory=list)
  raw_steps: list[dict] = field(default_factory=list)
  system_prompt: str = ''
  loaded_skills: list[str] = field(default_factory=list)


def _classify_tool_call(tool_name: str) -> ActionType:
  mapping = {
      'view_file': ActionType.FILE_VIEW,
      'write_to_file': ActionType.FILE_WRITE,
      'replace_file_content': ActionType.FILE_EDIT,
      'multi_replace_file_content': ActionType.FILE_EDIT,
      'run_command': ActionType.COMMAND_RUN,
      'command_status': ActionType.COMMAND_STATUS,
      'task_boundary': ActionType.TASK_BOUNDARY,
      'code_search': ActionType.CODE_SEARCH,
      'build_targets': ActionType.BUILD,
      'test_targets': ActionType.TEST,
      'notify_user': ActionType.NOTIFY_USER,
      'grep_search': ActionType.CODE_SEARCH,
      'find_by_name': ActionType.CODE_SEARCH,
      'build_cleaner': ActionType.BUILD,
  }
  return mapping.get(tool_name, ActionType.TOOL_CALL)


def _parse_step(step: dict, idx: int) -> list[AgentAction]:
  actions = []
  inner = step.get('step', {})
  timestamp = step.get('timestamp', '')

  if 'userInput' in inner:
    items = inner['userInput'].get('items', [])
    texts = [item.get('text', '') for item in items if item.get('text')]
    actions.append(
        AgentAction(
            action_type=ActionType.USER_INPUT,
            step_index=idx,
            timestamp=timestamp,
            content='\n'.join(texts),
        )
    )

  if 'plannerResponse' in inner:
    resp = inner['plannerResponse']
    response_text = resp.get('response', '')
    if response_text:
      actions.append(
          AgentAction(
              action_type=ActionType.MODEL_RESPONSE,
              step_index=idx,
              timestamp=timestamp,
              content=response_text,
          )
      )
    for tc in resp.get('toolCalls', []):
      tool_name = tc.get('name', '')
      try:
        args = json.loads(tc.get('argumentsJson', '{}'))
      except (json.JSONDecodeError, TypeError):
        args = {}
      action_type = _classify_tool_call(tool_name)
      file_path = ''
      if action_type == ActionType.FILE_VIEW:
        file_path = args.get('AbsolutePath', '')
      elif action_type in (ActionType.FILE_WRITE, ActionType.FILE_EDIT):
        file_path = args.get('TargetFile', '')
      actions.append(
          AgentAction(
              action_type=action_type,
              step_index=idx,
              timestamp=timestamp,
              tool_name=tool_name,
              arguments=args,
              file_path=file_path,
          )
      )

  if 'viewFile' in inner:
    vf = inner['viewFile']
    path = vf.get('absolutePathUri', '')
    is_skill = 'SKILL.md' in path or '/skills/' in path
    actions.append(
        AgentAction(
            action_type=ActionType.SKILL_READ
            if is_skill
            else ActionType.FILE_VIEW,
            step_index=idx,
            timestamp=timestamp,
            file_path=path,
        )
    )

  if 'runCommand' in inner:
    rc = inner['runCommand']
    exit_code = rc.get('exitCode', 0)
    actions.append(
        AgentAction(
            action_type=ActionType.COMMAND_RUN,
            step_index=idx,
            timestamp=timestamp,
            content=rc.get('commandLine', ''),
            success=exit_code == 0,
        )
    )

  if 'taskBoundary' in inner:
    tb = inner['taskBoundary']
    actions.append(
        AgentAction(
            action_type=ActionType.TASK_BOUNDARY,
            step_index=idx,
            timestamp=timestamp,
            content=f"{tb.get('taskName', '')} | {tb.get('taskStatus', '')}",
        )
    )

  if 'writeFile' in inner:
    wf = inner['writeFile']
    actions.append(
        AgentAction(
            action_type=ActionType.FILE_WRITE,
            step_index=idx,
            timestamp=timestamp,
            file_path=wf.get('absolutePathUri', ''),
        )
    )

  if 'replaceFileContent' in inner:
    rfc = inner['replaceFileContent']
    actions.append(
        AgentAction(
            action_type=ActionType.FILE_EDIT,
            step_index=idx,
            timestamp=timestamp,
            file_path=rfc.get('absolutePathUri', ''),
        )
    )

  if 'notifyUser' in inner:
    nu = inner['notifyUser']
    actions.append(
        AgentAction(
            action_type=ActionType.NOTIFY_USER,
            step_index=idx,
            timestamp=timestamp,
            content=nu.get('message', ''),
        )
    )

  return actions


def _extract_loaded_skills(system_prompt: str) -> list[str]:
  skills = []
  for match in re.finditer(
      r'- (\S+) \([^)]+/skills/\S+/SKILL\.md\)', system_prompt
  ):
    skills.append(match.group(1))
  return skills


def analyze_from_json(
    steps_json: dict, metadata: dict | None = None
) -> TrajectoryData:
  step_list = steps_json.get('steps', [])
  actions = []
  for step in step_list:
    idx = step.get('step_index', 0)
    actions.extend(_parse_step(step, idx))

  traj = TrajectoryData(
      trajectory_id=metadata.get('trajectory_id', '') if metadata else '',
      source='json',
      timestamp=metadata.get('timestamp', '') if metadata else '',
      num_steps=len(step_list),
      actions=actions,
      raw_steps=step_list,
  )
  return traj


def analyze_from_trajectory_cli(
    trajectory_id: str,
    trajectory_cli: str = '/google/bin/releases/gemini-agents-trajectory/trajectory_cli',
) -> TrajectoryData:
  result = subprocess.run(
      [
          trajectory_cli,
          f'--trajectory_id={trajectory_id}',
          '--skip_generations',
      ],
      capture_output=True,
      text=True,
      timeout=60,
  )
  if result.returncode != 0:
    raise RuntimeError(f'trajectory_cli failed: {result.stderr[:500]}')
  return TrajectoryData(
      trajectory_id=trajectory_id,
      source='cli',
      actions=[],
      raw_steps=[],
  )


def analyze_from_file(filepath: str) -> TrajectoryData:
  with open(filepath) as f:
    data = json.load(f)
  if 'steps' in data:
    return analyze_from_json(data)
  raise ValueError(f'Unrecognized trajectory format in {filepath}')


def get_action_summary(data: TrajectoryData) -> dict[str, int]:
  counts: dict[str, int] = {}
  for action in data.actions:
    key = action.action_type.value
    counts[key] = counts.get(key, 0) + 1
  return dict(sorted(counts.items(), key=lambda x: -x[1]))


def get_files_accessed(data: TrajectoryData) -> dict[str, list[str]]:
  files: dict[str, list[str]] = {'viewed': [], 'written': [], 'edited': []}
  seen = set()
  for action in data.actions:
    if not action.file_path:
      continue
    if (
        action.action_type == ActionType.FILE_VIEW
        and action.file_path not in seen
    ):
      files['viewed'].append(action.file_path)
      seen.add(action.file_path)
    elif action.action_type == ActionType.FILE_WRITE:
      files['written'].append(action.file_path)
    elif action.action_type == ActionType.FILE_EDIT:
      files['edited'].append(action.file_path)
  return files


def get_skill_reads(data: TrajectoryData) -> list[str]:
  return [
      a.file_path
      for a in data.actions
      if a.action_type == ActionType.SKILL_READ
  ]
