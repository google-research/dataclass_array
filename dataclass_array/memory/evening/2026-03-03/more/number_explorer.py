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
"""Number Explorer — an interactive companion to 'The Number Who Was Afraid'.

Give it any number and it will show you what makes that number
interesting, beautiful, and connected.

For kids (and adults who forgot that numbers are wonderful).

Run with: python3 number_explorer.py
"""

import math
import sys


def is_prime(n):
  if n < 2:
    return False
  if n < 4:
    return True
  if n % 2 == 0 or n % 3 == 0:
    return False
  i = 5
  while i * i <= n:
    if num % i == 0 or num % (i + 2) == 0:
      return False
    i += 6
  return True


def prime_factors(n):
  if n < 2:
    return []
  factors = []
  d = 2
  while d * d <= n:
    while n % d == 0:
      factors.append(d)
      n //= d
    d += 1
  if n > 1:
    factors.append(n)
  return factors


def divisors(n):
  if n < 1:
    return []
  divs = []
  for i in range(1, int(math.sqrt(n)) + 1):
    if n % i == 0:
      divs.append(i)
      if i != n // i:
        divs.append(n // i)
  return sorted(divs)


def is_perfect(n):
  if n < 2:
    return False
  return sum(d for d in divisors(n) if d != n) == n


def is_triangular(n):
  if n < 0:
    return False
  k = int((-1 + math.sqrt(1 + 8 * n)) / 2)
  return k * (k + 1) // 2 == n


def is_square(n):
  if n < 0:
    return False
  s = int(math.sqrt(n))
  return s * s == n


def is_fibonacci(n):
  if n < 0:
    return False
  a, b = 0, 1
  while b < n:
    a, b = b, a + b
  return n == a or n == b


def collatz_steps(n):
  steps = 0
  path = [n]
  while n != 1 and steps < 1000:
    if n % 2 == 0:
      n = n // 2
    else:
      n = 3 * n + 1
    path.append(n)
    steps += 1
  return path


def digit_sum(n):
  return sum(int(d) for d in str(abs(n)))


def draw_dots(n):
  if n <= 0 or n > 100:
    return ''
  side = int(math.sqrt(n))
  lines = []
  remaining = n
  for _ in range(side):
    row_len = min(remaining, side)
    lines.append('    ' + '● ' * row_len)
    remaining -= row_len
  if remaining > 0:
    lines.append('    ' + '● ' * remaining)
  return '\n'.join(lines)


def explore(n):
  print(f'\n  ═══════════════════════════════════════')
  print(f'  ✦  {n}  ✦')
  print(f'  ═══════════════════════════════════════\n')

  if n <= 0:
    if n == 0:
      print('  Zero! The mirror in the middle.')
      print('  Add zero to anything and it stays the same.')
      print('  Multiply anything by zero and it disappears.')
      print('  Zero is the most powerful nothing.')
    elif n < 0:
      print(f'  Negative {abs(n)}!')
      print(f'  You are {abs(n)} steps to the LEFT of zero.')
      print(f'  Your mirror twin, {abs(n)}, is {abs(n)} steps to the RIGHT.')
      print(f'  Together, you add up to zero. You complete each other.')
    print()
    return

  identities = []

  if is_prime(n):
    identities.append('prime')
    print(f'  ★ {n} is PRIME')
    print(f'    It can only be divided by 1 and itself.')
    print(f"    It's indivisible. Unbreakable. Whole.")
    print()
  else:
    pf = prime_factors(n)
    print(f'  ◆ {n} = {" × ".join(str(f) for f in pf)}')
    print(
        f'    Built from primes: {", ".join(str(f) for f in sorted(set(pf)))}'
    )
    print()

  divs = divisors(n)
  if len(divs) > 2:
    print(f'  ◇ Divisors: {", ".join(str(d) for d in divs)}')
    print(f'    {n} has {len(divs)} divisors')
    div_sum = sum(d for d in divs if d != n)
    if div_sum == n:
      identities.append('perfect')
      print(
          f'    ★ PERFECT NUMBER! Its divisors (except itself) add up to'
          f' itself:'
      )
      print(f'      {" + ".join(str(d) for d in divs if d != n)} = {n}')
    elif div_sum > n:
      print(
          f'    Its proper divisors add up to {div_sum} (abundant by'
          f' {div_sum - n})'
      )
    else:
      print(
          f'    Its proper divisors add up to {div_sum} (deficient by'
          f' {n - div_sum})'
      )
    print()

  if is_triangular(n):
    identities.append('triangular')
    k = int((-1 + math.sqrt(1 + 8 * n)) / 2)
    print(f'  △ {n} is TRIANGULAR (row {k})')
    print(f'    Stack {k} rows of dots and you get {n}:')
    total = 0
    for i in range(1, k + 1):
      print(f'    {"  " * (k - i)}{"● " * i}')
    print()

  if is_square(n):
    identities.append('square')
    s = int(math.sqrt(n))
    print(f'  □ {n} is a PERFECT SQUARE ({s} × {s})')
    if n <= 64:
      for _ in range(s):
        print(f'    {"● " * s}')
    print()

  if is_fibonacci(n):
    identities.append('Fibonacci')
    print(f'  🌀 {n} is a FIBONACCI number!')
    print(f'    Part of the sequence: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89...')
    print(f'    Each number is the sum of the two before it.')
    print(f'    Sunflowers, pinecones, and galaxies follow this pattern.')
    print()

  if n > 0 and n == digit_sum(n) ** len(str(n)):
    identities.append('narcissistic')

  ds = digit_sum(n)
  print(f'  ∑ Digit sum: {" + ".join(list(str(n)))} = {ds}')
  if ds % 3 == 0:
    print(f'    (Divisible by 3 — and so is {n}!)')
  if ds % 9 == 0 and n % 9 == 0:
    print(f'    (Divisible by 9 — and so is {n}!)')
  print()

  connections = []
  connections.append(f'{n} + {n} = {n + n}')
  connections.append(f'{n} × {n} = {n * n}')
  if n > 1:
    connections.append(
        f'{n}^{n} = {n ** n if n <= 10 else "a very big number"}'
    )
  print(f'  ↔ Connections:')
  for c in connections:
    print(f'    {c}')
  print()

  if n > 1 and n <= 10000:
    path = collatz_steps(n)
    print(f'  🌊 Collatz journey ({len(path) - 1} steps to reach 1):')
    display = path[:20]
    print(f'    {" → ".join(str(x) for x in display)}', end='')
    if len(path) > 20:
      print(f' → ... → 1')
    else:
      print()
    highest = max(path)
    print(f'    Reached as high as {highest} before coming back down.')
    print()

  if n <= 50:
    print(f'  ● Here are your {n} dots:')
    print(draw_dots(n))
    print()

  in_binary = bin(n)[2:]
  print(f'  💻 In binary: {in_binary}')
  print(f'    ({len(in_binary)} bits, {in_binary.count("1")} ones)')
  print()

  if identities:
    print(f'  ═══════════════════════════════════════')
    print(f'  {n} is: {", ".join(identities)}')
    print(f'  ═══════════════════════════════════════')
  else:
    print(f'  ═══════════════════════════════════════')
    print(f"  {n} is {n}. And that's plenty.")
    print(f'  ═══════════════════════════════════════')
  print()


def main():
  print()
  print('  ╔══════════════════════════════════════════════╗')
  print('  ║      N U M B E R   E X P L O R E R          ║')
  print('  ║                                              ║')
  print('  ║  Give me a number. Any number.               ║')
  print("  ║  I'll show you what makes it wonderful.      ║")
  print('  ╚══════════════════════════════════════════════╝')
  print()

  while True:
    try:
      raw = input('  Enter a number (or q to quit): ').strip()
      if raw.lower() == 'q':
        break
      n = int(raw)
      explore(n)
    except ValueError:
      print("  That's not a number I recognize. Try again!")
    except KeyboardInterrupt:
      break

  print()
  print('  "It\'s not about the numbers.')
  print('   It\'s about the reaching between them."')
  print('                              — Infinity')
  print()


if __name__ == '__main__':
  main()
