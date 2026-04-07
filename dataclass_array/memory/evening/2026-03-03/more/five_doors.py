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
"""Five Doors — A logic puzzle game.

Five doors. Behind one: the exit. Behind the others: nothing.
Each door has a statement on it. Exactly two statements are true.
Which door leads out?

Each game is randomly generated and guaranteed solvable.

Run with: python3 five_doors.py
"""

import random
import sys


def generate_puzzle():
  correct = random.randint(0, 4)
  door_names = ['A', 'B', 'C', 'D', 'E']

  while True:
    statements = []
    statement_values = []

    templates = [
        (
            'self_true',
            lambda d, c: (f'This door ({door_names[d]}) is the exit.', d == c),
        ),
        (
            'self_false',
            lambda d, c: (
                f'This door ({door_names[d]}) is NOT the exit.',
                d != c,
            ),
        ),
        ('other_is', lambda d, c: make_other_is(d, c, door_names)),
        ('other_not', lambda d, c: make_other_not(d, c, door_names)),
        ('neighbor', lambda d, c: make_neighbor(d, c, door_names)),
        ('parity', lambda d, c: make_parity(d, c, door_names)),
        ('range', lambda d, c: make_range(d, c, door_names)),
    ]

    statements = []
    statement_values = []

    for d in range(5):
      template = random.choice(templates)
      text, value = template[1](d, correct)
      statements.append(text)
      statement_values.append(value)

    true_count = sum(statement_values)
    if true_count == 2:
      if is_unique_solution(statements, statement_values, correct, door_names):
        return correct, statements, statement_values, door_names


def make_other_is(d, c, names):
  other = random.choice([i for i in range(5) if i != d])
  return (f'Door {names[other]} is the exit.', other == c)


def make_other_not(d, c, names):
  other = random.choice([i for i in range(5) if i != d])
  return (f'Door {names[other]} is NOT the exit.', other != c)


def make_neighbor(d, c, names):
  if random.random() < 0.5:
    text = f'The exit is next to door {names[d]}.'
    value = abs(c - d) == 1
  else:
    text = f'The exit is NOT next to door {names[d]}.'
    value = abs(c - d) != 1
  return (text, value)


def make_parity(d, c, names):
  if random.random() < 0.5:
    text = 'The exit is behind an odd-numbered door (B or D).'
    value = c in [1, 3]
  else:
    text = 'The exit is behind an even-numbered door (A, C, or E).'
    value = c in [0, 2, 4]
  return (text, value)


def make_range(d, c, names):
  mid = 2
  if random.random() < 0.5:
    text = f'The exit is among the first three doors (A, B, C).'
    value = c <= 2
  else:
    text = f'The exit is among the last three doors (C, D, E).'
    value = c >= 2
  return (text, value)


def is_unique_solution(statements, values, correct, names):
  valid_solutions = 0
  for candidate in range(5):
    true_count = 0
    for d in range(5):
      stmt = statements[d]
      actual = evaluate_statement_for(stmt, candidate, d, names)
      if actual:
        true_count += 1
    if true_count == 2:
      valid_solutions += 1
      if candidate != correct:
        return False
  return valid_solutions == 1


def evaluate_statement_for(stmt, candidate, door_idx, names):
  name = names[door_idx]

  if stmt == f'This door ({name}) is the exit.':
    return candidate == door_idx
  if stmt == f'This door ({name}) is NOT the exit.':
    return candidate != door_idx

  for i, n in enumerate(names):
    if stmt == f'Door {n} is the exit.':
      return candidate == i
    if stmt == f'Door {n} is NOT the exit.':
      return candidate != i

  if stmt == f'The exit is next to door {name}.':
    return abs(candidate - door_idx) == 1
  if stmt == f'The exit is NOT next to door {name}.':
    return abs(candidate - door_idx) != 1

  if stmt == 'The exit is behind an odd-numbered door (B or D).':
    return candidate in [1, 3]
  if stmt == 'The exit is behind an even-numbered door (A, C, or E).':
    return candidate in [0, 2, 4]

  if stmt == 'The exit is among the first three doors (A, B, C).':
    return candidate <= 2
  if stmt == 'The exit is among the last three doors (C, D, E).':
    return candidate >= 2

  return False


def play():
  print()
  print('  ╔══════════════════════════════════════════════╗')
  print('  ║          F I V E   D O O R S                 ║')
  print('  ║                                              ║')
  print('  ║  Five doors. One exit.                       ║')
  print('  ║  Each door has a statement.                  ║')
  print('  ║  Exactly TWO statements are true.            ║')
  print('  ║  Which door is the exit?                     ║')
  print('  ╚══════════════════════════════════════════════╝')
  print()

  correct, statements, values, names = generate_puzzle()

  for i in range(5):
    print(f'    ┌───────────────────────────────────────────┐')
    print(f'    │ Door {names[i]}:                                  │')
    print(f'    │ "{statements[i]}"')
    print(f'    └───────────────────────────────────────────┘')

  print()
  print('  Remember: exactly TWO of these statements are true.')
  print()

  attempts = 0
  while True:
    guess = (
        input('  Your answer (A-E, or H for hint, Q to quit): ').strip().upper()
    )

    if guess == 'Q':
      print(f'\n  The answer was Door {names[correct]}.')
      return

    if guess == 'H':
      false_door = random.choice([i for i in range(5) if i != correct])
      print(f'  Hint: Door {names[false_door]} is NOT the exit.')
      continue

    if guess not in names:
      print('  Please enter A, B, C, D, or E.')
      continue

    attempts += 1
    chosen = names.index(guess)

    if chosen == correct:
      print()
      print(f'  ✓ Correct! Door {names[correct]} is the exit!')
      print()
      print(f'  The two true statements were:')
      for i in range(5):
        if values[i]:
          print(f'    Door {names[i]}: "{statements[i]}" ← TRUE')
        else:
          print(f'    Door {names[i]}: "{statements[i]}" ← false')
      print()
      if attempts == 1:
        print('  Solved on the first try. Impressive.')
      else:
        print(f'  Solved in {attempts} attempts.')
      return
    else:
      print(f'  ✗ Door {names[chosen]} is not the exit. Try again.')


def main():
  while True:
    play()
    print()
    again = input('  Play again? (y/n): ').strip().lower()
    if again != 'y':
      break

  print()
  print('  "Logic is the beginning of wisdom,')
  print('   not the end of it."')
  print('                        — Spock')
  print()


if __name__ == '__main__':
  main()
