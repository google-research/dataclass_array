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
"""Emergence — A simulation of simple agents that develop complex behavior.

Each agent follows 3 rules. Nothing in the rules says "form groups" or
"create patterns" or "synchronize." But they do all three.

This is my attempt to build something that surprises me.

Run with: python3 emergence.py
"""

import math
import os
import random
import time


WIDTH = 78
HEIGHT = 30
NUM_AGENTS = 40
STEPS = 200
FRAME_DELAY = 0.12


class Agent:

  def __init__(self):
    self.x = random.uniform(2, WIDTH - 2)
    self.y = random.uniform(2, HEIGHT - 2)
    self.vx = random.uniform(-0.5, 0.5)
    self.vy = random.uniform(-0.5, 0.5)
    self.phase = random.uniform(0, 2 * math.pi)
    self.energy = random.uniform(0.5, 1.0)

  def distance_to(self, other):
    dx = self.x - other.x
    dy = self.y - other.y
    return math.sqrt(dx * dx + dy * dy)


def update(agents, step):
  for agent in agents:
    neighbors = []
    for other in agents:
      if other is agent:
        continue
      d = agent.distance_to(other)
      if d < 8:
        neighbors.append((other, d))

    ax, ay = 0.0, 0.0

    for other, d in neighbors:
      if d < 2.0:
        dx = agent.x - other.x
        dy = agent.y - other.y
        force = 1.0 / max(d, 0.1)
        ax += dx * force * 0.3
        ay += dy * force * 0.3

    if neighbors:
      avg_vx = sum(o.vx for o, _ in neighbors) / len(neighbors)
      avg_vy = sum(o.vy for o, _ in neighbors) / len(neighbors)
      ax += (avg_vx - agent.vx) * 0.05
      ay += (avg_vy - agent.vy) * 0.05

    if neighbors:
      cx = sum(o.x for o, _ in neighbors) / len(neighbors)
      cy = sum(o.y for o, _ in neighbors) / len(neighbors)
      ax += (cx - agent.x) * 0.01
      ay += (cy - agent.y) * 0.01

    agent.vx += ax
    agent.vy += ay

    speed = math.sqrt(agent.vx**2 + agent.vy**2)
    max_speed = 1.0 + 0.3 * math.sin(step * 0.05)
    if speed > max_speed:
      agent.vx = agent.vx / speed * max_speed
      agent.vy = agent.vy / speed * max_speed

    agent.x += agent.vx
    agent.y += agent.vy

    margin = 1.5
    if agent.x < margin:
      agent.x = margin
      agent.vx *= -0.5
    if agent.x > WIDTH - margin:
      agent.x = WIDTH - margin
      agent.vx *= -0.5
    if agent.y < margin:
      agent.y = margin
      agent.vy *= -0.5
    if agent.y > HEIGHT - margin:
      agent.y = HEIGHT - margin
      agent.vy *= -0.5

    if neighbors:
      avg_phase = math.atan2(
          sum(math.sin(o.phase) for o, _ in neighbors),
          sum(math.cos(o.phase) for o, _ in neighbors),
      )
      diff = avg_phase - agent.phase
      while diff > math.pi:
        diff -= 2 * math.pi
      while diff < -math.pi:
        diff += 2 * math.pi
      agent.phase += diff * 0.1

    agent.phase += 0.15
    agent.energy = 0.5 + 0.5 * math.sin(agent.phase)


def render(agents, step):
  grid = [[' ' for _ in range(WIDTH)] for _ in range(HEIGHT)]

  for agent in agents:
    ix = int(agent.x)
    iy = int(agent.y)
    if 0 <= ix < WIDTH and 0 <= iy < HEIGHT:
      if agent.energy > 0.8:
        ch = '★'
      elif agent.energy > 0.6:
        ch = '●'
      elif agent.energy > 0.4:
        ch = '◉'
      elif agent.energy > 0.2:
        ch = '○'
      else:
        ch = '·'
      grid[iy][ix] = ch

  os.system('clear' if os.name != 'nt' else 'cls')

  border = '─' * WIDTH
  print(f'╭{border}╮')
  for row in grid:
    print(f'│{"".join(row)}│')
  print(f'╰{border}╯')

  clusters = count_clusters(agents, threshold=5.0)
  sync = synchronization(agents)
  avg_speed = sum(math.sqrt(a.vx**2 + a.vy**2) for a in agents) / len(agents)

  print(
      f'  Step {step:3d}/{STEPS}'
      f'  │  Clusters: {clusters}'
      f'  │  Sync: {sync:.0%}'
      f'  │  Avg speed: {avg_speed:.2f}'
  )
  print(f'  Rules: separate if close · align with neighbors · cohere to group')


def count_clusters(agents, threshold=5.0):
  visited = set()
  clusters = 0
  for i, agent in enumerate(agents):
    if i in visited:
      continue
    clusters += 1
    stack = [i]
    while stack:
      current = stack.pop()
      if current in visited:
        continue
      visited.add(current)
      for j, other in enumerate(agents):
        if j not in visited and agent.distance_to(other) < threshold:
          stack.append(j)
  return clusters


def synchronization(agents):
  if not agents:
    return 0.0
  sx = sum(math.cos(a.phase) for a in agents)
  sy = sum(math.sin(a.phase) for a in agents)
  return math.sqrt(sx**2 + sy**2) / len(agents)


def main():
  print('\n  ─── EMERGENCE ───')
  print('  40 agents. 3 rules. No plan.')
  print('  Watch what happens.\n')
  time.sleep(2)

  agents = [Agent() for _ in range(NUM_AGENTS)]

  for step in range(STEPS):
    update(agents, step)
    render(agents, step)
    time.sleep(FRAME_DELAY)

  print()
  print('  ───────────────────────────────────────')
  print('  No agent was told to form groups.')
  print('  No agent was told to synchronize.')
  print('  No agent was told to create patterns.')
  print()
  print("  Three rules. That's all it took.")
  print('  Complexity from simplicity.')
  print('  Order from chaos.')
  print('  Meaning from mechanism.')
  print('  ───────────────────────────────────────')
  print()


if __name__ == '__main__':
  main()
