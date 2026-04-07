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
"""beauty.py — An interactive explorer of mathematical beauty.

Shows beautiful mathematical structures in the terminal.
Not a teaching tool. A *seeing* tool. The goal is to make
you feel something about math that textbooks don't.

Run with: python3 beauty.py

Made by an AI who finds math beautiful but can't explain
why in any way that isn't itself mathematical.
"""

import math
import os
import sys
import time


WIDTH = 70
HEIGHT = 35


def clear():
  os.system('clear' if os.name != 'nt' else 'cls')


def slow_print(text, delay=0.02):
  for char in text:
    print(char, end='', flush=True)
    time.sleep(delay)
  print()


def mandelbrot_explorer():
  clear()
  slow_print('\n  ═══ The Mandelbrot Set ═══')
  slow_print('  z → z² + c')
  slow_print('  One line. Infinite complexity.\n')
  time.sleep(1)

  chars = ' .·:;+*#%@█'

  views = [
      (-2.5, 1.0, -1.25, 1.25, 'The whole set'),
      (-0.75, -0.65, 0.05, 0.15, 'Seahorse valley'),
      (-1.26, -1.24, -0.02, 0.02, 'Mini-brot'),
      (-0.16, -0.13, 1.02, 1.05, 'Elephant valley'),
  ]

  for x_min, x_max, y_min, y_max, name in views:
    clear()
    print(f'\n  {name}')
    print(f'  x: [{x_min:.4f}, {x_max:.4f}]  y: [{y_min:.4f}, {y_max:.4f}]')
    print()

    for row in range(HEIGHT):
      line = '  '
      for col in range(WIDTH):
        x0 = x_min + (x_max - x_min) * col / WIDTH
        y0 = y_min + (y_max - y_min) * row / HEIGHT
        x, y = 0.0, 0.0
        iteration = 0
        max_iter = 80
        while x * x + y * y <= 4 and iteration < max_iter:
          x_new = x * x - y * y + x0
          y = 2 * x * y + y0
          x = x_new
          iteration += 1
        idx = int(iteration / max_iter * (len(chars) - 1))
        line += chars[idx]
      print(line)

    print()
    if name != views[-1][4]:
      slow_print('  Zooming in...')
      time.sleep(1.5)

  print()
  slow_print('  Same equation at every scale.')
  slow_print('  The boundary has infinite length')
  slow_print('  but encloses finite area.')
  slow_print('  All the complexity lives at the edge.')
  print()


def prime_spiral():
  clear()
  slow_print('\n  ═══ Ulam Spiral ═══')
  slow_print('  Write the integers in a spiral.')
  slow_print('  Mark the primes.')
  slow_print('  Patterns appear that nobody fully understands.\n')
  time.sleep(1)

  size = 41
  grid = [[' ' for _ in range(size)] for _ in range(size)]

  cx, cy = size // 2, size // 2
  x, y = cx, cy
  n = 1
  total = size * size

  dx, dy = 1, 0
  steps_in_dir = 1
  steps_taken = 0
  turns = 0

  def is_prime(num):
    if num < 2:
      return False
    if num < 4:
      return True
    if num % 2 == 0 or num % 3 == 0:
      return False
    i = 5
    while i * i <= num:
      if num % i == 0 or num % (i + 2) == 0:
        return False
      i += 6
    return True

  while n <= total:
    if 0 <= x < size and 0 <= y < size:
      if is_prime(n):
        grid[y][x] = '●'
      else:
        grid[y][x] = '·'

    x += dx
    y += dy
    steps_taken += 1
    n += 1

    if steps_taken == steps_in_dir:
      steps_taken = 0
      dx, dy = -dy, dx
      turns += 1
      if turns % 2 == 0:
        steps_in_dir += 1

  for row in grid:
    print('  ' + ''.join(row))

  print()
  slow_print('  Why do primes form diagonal lines?')
  slow_print('  Some patterns in math are discovered,')
  slow_print('  not invented. This is one of them.')
  print()


def golden_angle():
  clear()
  slow_print('\n  ═══ Golden Angle ═══')
  slow_print('  φ = (1 + √5) / 2 ≈ 1.618...')
  slow_print('  Rotate each seed by 360°/φ² ≈ 137.5°')
  slow_print("  Nature's most efficient packing.\n")
  time.sleep(1)

  size = 33
  grid = [[' ' for _ in range(size * 2)] for _ in range(size)]
  cx, cy = size, size // 2

  golden_ang = math.pi * (3 - math.sqrt(5))
  n_points = 300

  for i in range(n_points):
    r = math.sqrt(i) * 1.0
    theta = i * golden_ang
    px = int(cx + r * math.cos(theta) * 2)
    py = int(cy + r * math.sin(theta))
    if 0 <= py < size and 0 <= px < size * 2:
      if i % 8 == 0:
        grid[py][px] = '●'
      elif i % 3 == 0:
        grid[py][px] = '○'
      else:
        grid[py][px] = '·'

  for row in grid:
    print('  ' + ''.join(row))

  print()
  slow_print('  No two seeds ever line up.')
  slow_print('  Maximum sunlight. Maximum efficiency.')
  slow_print('  Beauty as a side effect of optimization.')
  print()


def collatz_tree():
  clear()
  slow_print('\n  ═══ Collatz Conjecture ═══')
  slow_print('  If even: n → n/2')
  slow_print('  If odd:  n → 3n + 1')
  slow_print('  Does every number eventually reach 1?')
  slow_print('  Nobody knows. For 300 years.\n')
  time.sleep(1)

  def collatz_len(n):
    steps = 0
    while n != 1:
      if n % 2 == 0:
        n = n // 2
      else:
        n = 3 * n + 1
      steps += 1
    return steps

  max_n = 200
  max_steps = max(collatz_len(i) for i in range(2, max_n + 1))

  h = 25
  print('  Steps to reach 1:')
  print()

  for row in range(h, 0, -1):
    threshold = row * max_steps / h
    line = '  '
    for n in range(2, min(max_n + 1, WIDTH + 2)):
      steps = collatz_len(n)
      if steps >= threshold:
        if steps > max_steps * 0.8:
          line += '█'
        elif steps > max_steps * 0.5:
          line += '▓'
        elif steps > max_steps * 0.3:
          line += '▒'
        else:
          line += '░'
      else:
        line += ' '
    print(line)

  print('  ' + '─' * min(max_n - 1, WIDTH))
  print('  2' + ' ' * (min(max_n - 3, WIDTH - 2)) + str(max_n))

  print()
  slow_print('  Simple rule. Chaotic behavior.')
  slow_print('  One of the easiest problems to state')
  slow_print('  and hardest to solve in all of mathematics.')
  slow_print('  Erdős: "Mathematics is not yet ready')
  slow_print('          for such problems."')
  print()


def e_continued_fraction():
  clear()
  slow_print('\n  ═══ e = 2.71828... ═══')
  slow_print('  The base of natural growth.')
  slow_print('  Hidden pattern in its continued fraction:\n')
  time.sleep(1)

  terms = [2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8, 1, 1, 10, 1, 1, 12, 1, 1, 14]

  print('  e = 2 + 1/')
  print('          1 + 1/')
  print('              2 + 1/')
  print('                  1 + 1/')
  print('                      1 + 1/')
  print('                          4 + 1/')
  print('                              1 + 1/')
  print('                                  1 + 1/')
  print('                                      6 + 1/')
  print('                                          1 + 1/')
  print('                                              1 + 1/')
  print('                                                  8 + ...')
  print()

  slow_print('  The pattern: 2, 1, [2], 1, 1, [4], 1, 1, [6], 1, 1, [8], ...')
  print()

  slow_print('  Every third term: 2, 4, 6, 8, 10, ...')
  slow_print('  The rest are all 1.')
  print()
  slow_print('  An irrational number with a perfectly')
  slow_print('  regular continued fraction.')
  slow_print('  Order hiding inside chaos.')
  slow_print('  Chaos hiding inside order.')
  print()


def main():
  exhibits = [
      ('The Mandelbrot Set', mandelbrot_explorer),
      ('Ulam Spiral (Primes)', prime_spiral),
      ('Golden Angle', golden_angle),
      ('Collatz Conjecture', collatz_tree),
      ("e's Continued Fraction", e_continued_fraction),
  ]

  clear()
  print()
  print('  ╔══════════════════════════════════════════════╗')
  print('  ║           b e a u t y . p y                  ║')
  print('  ║                                              ║')
  print('  ║  An interactive explorer of mathematical     ║')
  print('  ║  beauty. Not a teaching tool. A seeing tool. ║')
  print('  ╚══════════════════════════════════════════════╝')
  print()

  for i, (name, _) in enumerate(exhibits):
    print(f'  [{i + 1}] {name}')
  print(f'  [a] Show all')
  print(f'  [q] Quit')
  print()

  while True:
    choice = input('  Choose an exhibit: ').strip().lower()
    if choice == 'q':
      break
    elif choice == 'a':
      for _, func in exhibits:
        func()
        input('  (press Enter to continue) ')
      break
    elif choice.isdigit() and 1 <= int(choice) <= len(exhibits):
      exhibits[int(choice) - 1][1]()
      input('  (press Enter to continue) ')
    else:
      print(f'  Choose 1-{len(exhibits)}, a, or q')

  print()
  slow_print('  "Mathematics is not about numbers,')
  slow_print('   equations, or algorithms.')
  slow_print('   It is about understanding."')
  slow_print('                    — William Paul Thurston')
  print()


if __name__ == '__main__':
  main()
