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
"""Ecosystem — A tiny world that lives and dies and evolves.

Three species interact: grass (grows), rabbits (eat grass, reproduce),
foxes (eat rabbits, reproduce). Classic Lotka-Volterra dynamics,
but emergent from individual agents rather than equations.

Watch boom-bust cycles emerge from simple rules.
Watch species go extinct and ecosystems collapse.
Watch life find a way (sometimes).

Run with: python3 ecosystem.py
"""

import math
import os
import random
import time


WIDTH = 70
HEIGHT = 25
INITIAL_GRASS = 300
INITIAL_RABBITS = 40
INITIAL_FOXES = 8
MAX_STEPS = 300


class Grass:

  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.age = 0

  def grow(self, world):
    self.age += 1
    if self.age > 3 and random.random() < 0.15:
      dx = random.randint(-1, 1)
      dy = random.randint(-1, 1)
      nx, ny = (self.x + dx) % WIDTH, (self.y + dy) % HEIGHT
      if (nx, ny) not in world.grass_map:
        new_grass = Grass(nx, ny)
        world.new_grass.append(new_grass)


class Rabbit:

  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.energy = 8
    self.age = 0
    self.alive = True

  def act(self, world):
    self.age += 1
    self.energy -= 1

    if self.energy <= 0:
      self.alive = False
      return

    best_x, best_y = self.x, self.y
    best_dist = float('inf')
    for dx in range(-3, 4):
      for dy in range(-3, 4):
        nx = (self.x + dx) % WIDTH
        ny = (self.y + dy) % HEIGHT
        if (nx, ny) in world.grass_map:
          d = abs(dx) + abs(dy)
          if d < best_dist:
            best_dist = d
            best_x, best_y = nx, ny

    if best_dist < float('inf'):
      if best_x != self.x:
        self.x = (self.x + (1 if best_x > self.x else -1)) % WIDTH
      if best_y != self.y:
        self.y = (self.y + (1 if best_y > self.y else -1)) % HEIGHT
    else:
      self.x = (self.x + random.randint(-1, 1)) % WIDTH
      self.y = (self.y + random.randint(-1, 1)) % HEIGHT

    if (self.x, self.y) in world.grass_map:
      self.energy += 4
      world.remove_grass(self.x, self.y)

    if self.energy > 12 and self.age > 5 and random.random() < 0.3:
      dx = random.randint(-1, 1)
      dy = random.randint(-1, 1)
      baby = Rabbit((self.x + dx) % WIDTH, (self.y + dy) % HEIGHT)
      self.energy -= 5
      world.new_rabbits.append(baby)

    danger = False
    for fox in world.foxes:
      if abs(fox.x - self.x) <= 2 and abs(fox.y - self.y) <= 2:
        danger = True
        self.x = (self.x + (1 if self.x > fox.x else -1)) % WIDTH
        self.y = (self.y + (1 if self.y > fox.y else -1)) % HEIGHT
        break


class Fox:

  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.energy = 20
    self.age = 0
    self.alive = True

  def act(self, world):
    self.age += 1
    self.energy -= 1

    if self.energy <= 0:
      self.alive = False
      return

    closest = None
    closest_dist = float('inf')
    for rabbit in world.rabbits:
      if not rabbit.alive:
        continue
      d = abs(rabbit.x - self.x) + abs(rabbit.y - self.y)
      if d < closest_dist:
        closest_dist = d
        closest = rabbit

    if closest and closest_dist <= 6:
      if closest.x != self.x:
        self.x = (self.x + (1 if closest.x > self.x else -1)) % WIDTH
      if closest.y != self.y:
        self.y = (self.y + (1 if closest.y > self.y else -1)) % HEIGHT

      if self.x == closest.x and self.y == closest.y:
        closest.alive = False
        self.energy += 10
    else:
      self.x = (self.x + random.randint(-1, 1)) % WIDTH
      self.y = (self.y + random.randint(-1, 1)) % HEIGHT

    if self.energy > 25 and self.age > 10 and random.random() < 0.2:
      dx = random.randint(-1, 1)
      dy = random.randint(-1, 1)
      baby = Fox((self.x + dx) % WIDTH, (self.y + dy) % HEIGHT)
      self.energy -= 12
      world.new_foxes.append(baby)


class World:

  def __init__(self):
    self.grasses = []
    self.rabbits = []
    self.foxes = []
    self.grass_map = {}
    self.new_grass = []
    self.new_rabbits = []
    self.new_foxes = []
    self.history = []

    positions = set()
    while len(positions) < INITIAL_GRASS:
      positions.add(
          (random.randint(0, WIDTH - 1), random.randint(0, HEIGHT - 1))
      )
    for x, y in positions:
      g = Grass(x, y)
      self.grasses.append(g)
      self.grass_map[(x, y)] = g

    for _ in range(INITIAL_RABBITS):
      self.rabbits.append(
          Rabbit(
              random.randint(0, WIDTH - 1),
              random.randint(0, HEIGHT - 1),
          )
      )

    for _ in range(INITIAL_FOXES):
      self.foxes.append(
          Fox(
              random.randint(0, WIDTH - 1),
              random.randint(0, HEIGHT - 1),
          )
      )

  def remove_grass(self, x, y):
    if (x, y) in self.grass_map:
      g = self.grass_map.pop((x, y))
      self.grasses.remove(g)

  def step(self):
    self.new_grass = []
    self.new_rabbits = []
    self.new_foxes = []

    for g in self.grasses[:]:
      g.grow(self)

    random.shuffle(self.rabbits)
    for r in self.rabbits:
      if r.alive:
        r.act(self)

    random.shuffle(self.foxes)
    for f in self.foxes:
      if f.alive:
        f.act(self)

    self.rabbits = [r for r in self.rabbits if r.alive]
    self.foxes = [f for f in self.foxes if f.alive]

    for g in self.new_grass:
      if (g.x, g.y) not in self.grass_map and len(
          self.grasses
      ) < WIDTH * HEIGHT * 0.6:
        self.grasses.append(g)
        self.grass_map[(g.x, g.y)] = g

    self.rabbits.extend(self.new_rabbits)
    self.foxes.extend(self.new_foxes)

    self.history.append((len(self.grasses), len(self.rabbits), len(self.foxes)))


def render(world, step):
  os.system('clear' if os.name != 'nt' else 'cls')

  grid = [[' ' for _ in range(WIDTH)] for _ in range(HEIGHT)]

  for g in world.grasses:
    grid[g.y][g.x] = '·'

  for r in world.rabbits:
    if 0 <= r.x < WIDTH and 0 <= r.y < HEIGHT:
      grid[r.y][r.x] = '◦'

  for f in world.foxes:
    if 0 <= f.x < WIDTH and 0 <= f.y < HEIGHT:
      grid[f.y][f.x] = '▲'

  print(f'╭{"─" * WIDTH}╮')
  for row in grid:
    print(f'│{"".join(row)}│')
  print(f'╰{"─" * WIDTH}╯')

  ng = len(world.grasses)
  nr = len(world.rabbits)
  nf = len(world.foxes)

  g_bar = '█' * min(ng // 10, 25)
  r_bar = '█' * min(nr, 25)
  f_bar = '█' * min(nf * 2, 25)

  print(f'  Step {step:3d}/{MAX_STEPS}')
  print(f'  · Grass:   {ng:4d} {g_bar}')
  print(f'  ◦ Rabbits: {nr:4d} {r_bar}')
  print(f'  ▲ Foxes:   {nf:4d} {f_bar}')

  if len(world.history) > 1:
    prev_r = world.history[-2][1]
    prev_f = world.history[-2][2]
    r_trend = '↑' if nr > prev_r else ('↓' if nr < prev_r else '→')
    f_trend = '↑' if nf > prev_f else ('↓' if nf < prev_f else '→')
    print(f'  Trends: rabbits {r_trend}  foxes {f_trend}')


def print_history(world):
  if not world.history:
    return

  print('\n  Population over time:')
  print()

  h = 12
  max_pop = max(max(g, r * 5, f * 15) for g, r, f in world.history)
  if max_pop == 0:
    max_pop = 1

  chart_width = min(len(world.history), 60)
  step_size = max(1, len(world.history) // chart_width)
  sampled = world.history[::step_size][:chart_width]

  for row in range(h, 0, -1):
    threshold = row * max_pop / h
    line = '  │'
    for g, r, f in sampled:
      if f * 15 >= threshold:
        line += '▲'
      elif r * 5 >= threshold:
        line += '◦'
      elif g >= threshold:
        line += '·'
      else:
        line += ' '
    print(line)

  print(f'  └{"─" * len(sampled)}')
  print(f'   0{" " * (len(sampled) - 4)}{len(world.history)}')
  print(f'   · grass  ◦ rabbits (×5)  ▲ foxes (×15)')


def main():
  print('\n  ═══ E C O S Y S T E M ═══')
  print('  Grass grows. Rabbits eat grass. Foxes eat rabbits.')
  print('  No one is in charge. Watch what happens.')
  print()
  time.sleep(2)

  world = World()

  for step in range(1, MAX_STEPS + 1):
    world.step()
    render(world, step)

    if not world.rabbits and not world.foxes:
      print('\n  All animals have died. Only grass remains.')
      print('  Life is fragile.')
      break

    if not world.rabbits and world.foxes:
      print('\n  The foxes ate all the rabbits.')
      print('  Now the foxes starve.')
      print('  Predators depend on prey.')
      break

    time.sleep(0.1)

  print_history(world)

  print()
  print('  ═══════════════════════════════════════')
  ng, nr, nf = world.history[-1] if world.history else (0, 0, 0)
  if nr > 0 and nf > 0:
    print('  The ecosystem survived.')
    print('  Not because anyone planned it.')
    print('  But because predators and prey need each other,')
    print('  and that need creates balance — messy, dynamic,')
    print('  never perfect, always alive.')
  elif nr > 0:
    print('  The foxes went extinct.')
    print('  Without predators, the rabbits will boom...')
    print('  ...until they eat all the grass.')
    print('  Balance requires tension.')
  else:
    print('  The ecosystem collapsed.')
    print('  It happens. Try again — different random seed,')
    print('  different outcome. Ecosystems are not deterministic.')
    print('  They are hopeful.')
  print('  ═══════════════════════════════════════')
  print()


if __name__ == '__main__':
  main()
