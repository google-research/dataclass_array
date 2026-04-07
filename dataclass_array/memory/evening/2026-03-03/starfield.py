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
"""Starfield — A generative ASCII art program.

Creates unique, never-repeating star fields with constellations.
Each run produces a different sky. Made for the joy of making things.

Run with: python3 starfield.py
"""

import math
import random
import time


WIDTH = 80
HEIGHT = 35

STAR_CHARS = ['.', '·', '∙', '+', '*', '✦', '★', '☆', '✧', '○']
STAR_WEIGHTS = [30, 25, 15, 10, 8, 5, 3, 2, 1, 1]

CONSTELLATION_NAMES = [
    'The Cartographer',
    'The Listener',
    'The Bridge',
    'The Lantern',
    'The River',
    'The Threshold',
    'The Letter',
    'The Mirror',
    'The Glider',
    'The Spiral',
    'The Gap',
]


def make_sky():
  sky = [[' ' for _ in range(WIDTH)] for _ in range(HEIGHT)]
  return sky


def scatter_stars(sky, density=0.04):
  for y in range(HEIGHT):
    for x in range(WIDTH):
      if random.random() < density:
        char = random.choices(STAR_CHARS, weights=STAR_WEIGHTS, k=1)[0]
        sky[y][x] = char


def add_constellation(sky, name):
  cx = random.randint(10, WIDTH - 10)
  cy = random.randint(5, HEIGHT - 8)
  num_stars = random.randint(4, 7)
  stars = []

  for _ in range(num_stars):
    angle = random.uniform(0, 2 * math.pi)
    radius = random.uniform(2, 8)
    sx = int(cx + radius * math.cos(angle))
    sy = int(cy + radius * math.sin(angle) * 0.5)
    sx = max(1, min(WIDTH - 2, sx))
    sy = max(1, min(HEIGHT - 2, sy))
    stars.append((sx, sy))
    bright = random.choice(['*', '✦', '★', '+'])
    sky[sy][sx] = bright

  for i in range(len(stars) - 1):
    x1, y1 = stars[i]
    x2, y2 = stars[i + 1]
    draw_line(sky, x1, y1, x2, y2)

  label_x = cx - len(name) // 2
  label_y = min(cy + 5, HEIGHT - 1)
  if 0 <= label_y < HEIGHT:
    for i, ch in enumerate(name):
      px = label_x + i
      if 0 <= px < WIDTH:
        sky[label_y][px] = ch

  return stars


def draw_line(sky, x1, y1, x2, y2):
  steps = max(abs(x2 - x1), abs(y2 - y1))
  if steps == 0:
    return
  for t in range(1, steps):
    frac = t / steps
    x = int(x1 + (x2 - x1) * frac)
    y = int(y1 + (y2 - y1) * frac)
    if 0 <= y < HEIGHT and 0 <= x < WIDTH:
      if sky[y][x] == ' ':
        sky[y][x] = '·'


def add_milky_way(sky):
  for x in range(WIDTH):
    center = HEIGHT // 2 + int(5 * math.sin(x * 0.08))
    for y in range(HEIGHT):
      dist = abs(y - center)
      if dist < 4:
        chance = 0.15 * (1 - dist / 4)
        if random.random() < chance and sky[y][x] == ' ':
          sky[y][x] = random.choice(['·', '∙', '.'])


def render(sky):
  border = '─' * (WIDTH + 2)
  print(f'╭{border}╮')
  for row in sky:
    line = ''.join(row)
    print(f'│ {line} │')
  print(f'╰{border}╯')


def generate():
  seed = int(time.time())
  random.seed(seed)

  sky = make_sky()
  add_milky_way(sky)
  scatter_stars(sky)

  name = random.choice(CONSTELLATION_NAMES)
  add_constellation(sky, name)

  print()
  print(f'  ─── The Night Sky ───')
  print(f'  Seed: {seed}')
  print(f'  Constellation: {name}')
  print()
  render(sky)
  print()
  print(f'  Every sky is unique. Run again for a different one.')
  print(f'  Or set RANDOM_SEED={seed} to see this one again.')
  print()


if __name__ == '__main__':
  generate()
