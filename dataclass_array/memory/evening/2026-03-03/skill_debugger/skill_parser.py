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

"""Parses SKILL.md files into structured data for matching against trajectories.

Extracts:
- Skill metadata (name, description, triggers)
- CLI commands and their flags
- Key instructions (imperative sentences)
- File paths referenced
- Cross-references to other skills
"""

from dataclasses import dataclass, field
import os
import re


@dataclass
class CLICommand:
  binary_path: str
  flags: list[str] = field(default_factory=list)
  example_invocations: list[str] = field(default_factory=list)


@dataclass
class ParsedSkill:
  name: str
  description: str
  path: str
  triggers: list[str] = field(default_factory=list)
  cli_commands: list[CLICommand] = field(default_factory=list)
  key_instructions: list[str] = field(default_factory=list)
  referenced_paths: list[str] = field(default_factory=list)
  cross_references: list[str] = field(default_factory=list)
  content_lines: int = 0


def _extract_frontmatter(content: str) -> dict[str, str]:
  match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
  if not match:
    return {}
  fm = {}
  current_key = None
  current_val_lines = []
  for line in match.group(1).splitlines():
    kv = re.match(r'^(\w+):\s*(.*)', line)
    if kv:
      if current_key:
        fm[current_key] = ' '.join(current_val_lines).strip()
      current_key = kv.group(1)
      current_val_lines = [kv.group(2).lstrip('>').strip()]
    elif current_key and line.strip():
      current_val_lines.append(line.strip())
  if current_key:
    fm[current_key] = ' '.join(current_val_lines).strip()
  return fm


def _extract_triggers(description: str) -> list[str]:
  triggers = []
  trigger_phrases = re.findall(
      r'(?:Use when|Triggers? on|Use this skill when|Don\'t use for)'
      r'\s+["\']?(.+?)["\']?(?:\.|$)',
      description,
      re.IGNORECASE,
  )
  for phrase in trigger_phrases:
    parts = re.split(r'[,"]|"\s*,\s*"|\s+or\s+', phrase)
    for p in parts:
      p = p.strip().strip('"\'')
      if p and len(p) > 2:
        triggers.append(p.lower())
  return triggers


def _extract_cli_commands(content: str) -> list[CLICommand]:
  commands = []
  code_blocks = re.findall(r'```(?:bash|sh)\n(.*?)```', content, re.DOTALL)
  for block in code_blocks:
    lines = block.strip().splitlines()
    for line in lines:
      line = line.strip()
      if not line or line.startswith('#'):
        continue
      binary_match = re.match(
          r'(?:\$\w+|blaze\s+(?:run|build|test)|/google/bin/\S+)', line
      )
      if binary_match:
        binary = binary_match.group(0)
        flags = re.findall(r'--(\w+)', line)
        cmd = CLICommand(
            binary_path=binary,
            flags=flags,
            example_invocations=[line],
        )
        commands.append(cmd)
  return commands


def _extract_instructions(content: str) -> list[str]:
  body = re.sub(r'^---.*?---', '', content, flags=re.DOTALL).strip()
  body = re.sub(r'```.*?```', '', body, flags=re.DOTALL)
  instructions = []
  for line in body.splitlines():
    line = line.strip().lstrip('-*•').strip()
    if not line or len(line) < 10:
      continue
    if line.startswith('#'):
      continue
    if re.match(
        r'^(Use|Do not|Never|Always|Must|Should|Prefer|Avoid|Check|Run|Set|'
        r'Make sure|Ensure|Before|After|First|When|If you)',
        line,
        re.IGNORECASE,
    ):
      instructions.append(line)
  return instructions[:50]


def _extract_paths(content: str) -> list[str]:
  paths = set()
  for match in re.finditer(r'(?:/[\w._-]+){2,}[\w/._-]+', content):
    paths.add(match.group(0))
  return sorted(paths)


def _extract_cross_refs(content: str) -> list[str]:
  refs = set()
  for match in re.finditer(r'`([a-z][\w-]+)`', content):
    name = match.group(1)
    if len(name) > 2 and '-' in name:
      refs.add(name)
  return sorted(refs)


def parse_skill_file(skill_md_path: str) -> ParsedSkill | None:
  if not os.path.isfile(skill_md_path):
    return None
  with open(skill_md_path, 'r') as f:
    content = f.read()

  fm = _extract_frontmatter(content)
  name = fm.get('name', os.path.basename(os.path.dirname(skill_md_path)))
  description = fm.get('description', '')

  return ParsedSkill(
      name=name,
      description=description,
      path=skill_md_path,
      triggers=_extract_triggers(description),
      cli_commands=_extract_cli_commands(content),
      key_instructions=_extract_instructions(content),
      referenced_paths=_extract_paths(content),
      cross_references=_extract_cross_refs(content),
      content_lines=len(content.splitlines()),
  )


def parse_skill_directory(skills_root: str) -> list[ParsedSkill]:
  skills = []
  if not os.path.isdir(skills_root):
    return skills
  for entry in sorted(os.listdir(skills_root)):
    skill_dir = os.path.join(skills_root, entry)
    if not os.path.isdir(skill_dir):
      continue
    skill_md = os.path.join(skill_dir, 'SKILL.md')
    parsed = parse_skill_file(skill_md)
    if parsed:
      skills.append(parsed)
  return skills
