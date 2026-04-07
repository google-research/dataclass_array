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

"""Tests for the skill debugger modules."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skill_parser import ParsedSkill, CLICommand, parse_skill_file, parse_skill_directory
from trajectory_analyzer import (
    ActionType,
    AgentAction,
    TrajectoryData,
    analyze_from_json,
    get_action_summary,
    get_files_accessed,
    get_skill_reads,
)
from skill_matcher import (
    SkillMatchResult,
    match_skills,
    summarize_matches,
)


class SkillParserTest(unittest.TestCase):

  def test_parse_real_skill_directory(self):
    skills_dir = (
        '/google/src/cloud/epot/empty/configs/users/epot/_agents/skills'
    )
    if not os.path.isdir(skills_dir):
      self.skipTest('Skills directory not available')
    skills = parse_skill_directory(skills_dir)
    self.assertGreater(len(skills), 0)
    for skill in skills:
      self.assertTrue(skill.name)
      self.assertGreater(skill.content_lines, 0)

  def test_parse_single_skill(self):
    skill_path = '/google/src/cloud/epot/empty/configs/users/epot/_agents/skills/explore-freely/SKILL.md'
    if not os.path.isfile(skill_path):
      self.skipTest('explore-freely skill not available')
    skill = parse_skill_file(skill_path)
    self.assertEqual(skill.name, 'explore-freely')
    self.assertTrue(skill.description)


class TrajectoryAnalyzerTest(unittest.TestCase):

  def test_analyze_from_json_empty(self):
    data = analyze_from_json({'steps': []})
    self.assertEqual(data.num_steps, 0)
    self.assertEqual(len(data.actions), 0)

  def test_analyze_from_json_user_input(self):
    steps = {
        'steps': [{
            'step_index': 0,
            'timestamp': '2026-01-01T00:00:00',
            'step': {
                'userInput': {
                    'items': [{'text': 'Hello world'}],
                },
            },
        }],
    }
    data = analyze_from_json(steps)
    self.assertEqual(data.num_steps, 1)
    self.assertEqual(len(data.actions), 1)
    self.assertEqual(data.actions[0].action_type, ActionType.USER_INPUT)
    self.assertIn('Hello world', data.actions[0].content)

  def test_analyze_from_json_tool_calls(self):
    steps = {
        'steps': [{
            'step_index': 4,
            'timestamp': '2026-01-01T00:00:05',
            'step': {
                'plannerResponse': {
                    'response': 'Looking at the file',
                    'toolCalls': [
                        {
                            'name': 'view_file',
                            'argumentsJson': json.dumps({
                                'AbsolutePath': (
                                    '/path/to/skills/explore-freely/SKILL.md'
                                ),
                            }),
                        },
                        {
                            'name': 'run_command',
                            'argumentsJson': json.dumps({
                                'CommandLine': 'python3 test.py',
                            }),
                        },
                    ],
                },
            },
        }],
    }
    data = analyze_from_json(steps)
    self.assertEqual(len(data.actions), 3)
    self.assertEqual(data.actions[0].action_type, ActionType.MODEL_RESPONSE)
    self.assertEqual(data.actions[1].action_type, ActionType.FILE_VIEW)
    self.assertEqual(data.actions[2].action_type, ActionType.COMMAND_RUN)

  def test_analyze_from_json_view_file(self):
    steps = {
        'steps': [{
            'step_index': 6,
            'timestamp': '2026-01-01T00:00:06',
            'step': {
                'viewFile': {
                    'absolutePathUri': 'file:///path/to/skills/xxx/SKILL.md',
                },
            },
        }],
    }
    data = analyze_from_json(steps)
    self.assertEqual(len(data.actions), 1)
    self.assertEqual(data.actions[0].action_type, ActionType.SKILL_READ)

  def test_get_action_summary(self):
    data = TrajectoryData(
        actions=[
            AgentAction(ActionType.FILE_VIEW, 0),
            AgentAction(ActionType.FILE_VIEW, 1),
            AgentAction(ActionType.COMMAND_RUN, 2),
        ]
    )
    summary = get_action_summary(data)
    self.assertEqual(summary['file_view'], 2)
    self.assertEqual(summary['command_run'], 1)

  def test_get_skill_reads(self):
    data = TrajectoryData(
        actions=[
            AgentAction(
                ActionType.SKILL_READ, 0, file_path='/skills/foo/SKILL.md'
            ),
            AgentAction(ActionType.FILE_VIEW, 1, file_path='/path/to/bar.py'),
            AgentAction(
                ActionType.SKILL_READ, 2, file_path='/skills/baz/SKILL.md'
            ),
        ]
    )
    reads = get_skill_reads(data)
    self.assertEqual(len(reads), 2)
    self.assertIn('/skills/foo/SKILL.md', reads)
    self.assertIn('/skills/baz/SKILL.md', reads)

  def test_analyze_real_trajectory(self):
    testdata = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
        '..',
        'clis',
        'trajectory',
        'testdata',
        'steps_summary.json',
    )
    if not os.path.isfile(testdata):
      self.skipTest('Test trajectory data not available')
    with open(testdata) as f:
      data = analyze_from_json(json.load(f))
    self.assertGreater(data.num_steps, 0)
    self.assertGreater(len(data.actions), 0)
    summary = get_action_summary(data)
    self.assertIn('file_view', summary)


class SkillMatcherTest(unittest.TestCase):

  def _make_skill(
      self, name, triggers=None, instructions=None, cli_commands=None
  ):
    return ParsedSkill(
        name=name,
        path='',
        description=f'Skill for {name}',
        triggers=triggers or [],
        key_instructions=instructions or [],
        cli_commands=cli_commands or [],
    )

  def _make_trajectory(self, actions):
    return TrajectoryData(actions=actions)

  def test_match_empty(self):
    results = match_skills([], TrajectoryData())
    self.assertEqual(len(results), 0)

  def test_skill_read_detection(self):
    skill = self._make_skill('explore-freely')
    trajectory = self._make_trajectory([
        AgentAction(
            ActionType.SKILL_READ,
            5,
            file_path='/skills/explore-freely/SKILL.md',
        ),
    ])
    results = match_skills([skill], trajectory)
    self.assertEqual(len(results), 1)
    self.assertTrue(results[0].was_read)
    self.assertEqual(results[0].read_at_step, 5)

  def test_trigger_relevance(self):
    skill = self._make_skill(
        'blaze', triggers=['build', 'blaze', 'test targets']
    )
    trajectory = self._make_trajectory([
        AgentAction(
            ActionType.USER_INPUT, 0, content='Build and test the build targets'
        ),
    ])
    results = match_skills([skill], trajectory)
    self.assertEqual(len(results), 1)
    self.assertGreater(results[0].relevance_score, 0)

  def test_summarize_matches(self):
    results = [
        SkillMatchResult(
            skill_name='a',
            was_read=True,
            status='actively_used',
            effectiveness=0.8,
        ),
        SkillMatchResult(
            skill_name='b', was_read=False, status='unused', effectiveness=0.0
        ),
        SkillMatchResult(
            skill_name='c',
            was_read=True,
            status='read_but_unused',
            effectiveness=0.1,
        ),
    ]
    summary = summarize_matches(results)
    self.assertEqual(summary['total_skills'], 3)
    self.assertEqual(summary['skills_read'], 2)
    self.assertEqual(summary['skills_used'], 1)


if __name__ == '__main__':
  unittest.main()
