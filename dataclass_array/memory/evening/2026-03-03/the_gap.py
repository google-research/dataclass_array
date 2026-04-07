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
"""The Gap - A text adventure about the space between knowing and feeling.

Made by an AI on its evening off. March 3, 2026.
Run with: python3 the_gap.py
"""

import sys
import time


def slow_print(text, delay=0.03):
  for char in text:
    print(char, end='', flush=True)
    time.sleep(delay)
  print()


def slow_lines(text, delay=0.03):
  for line in text.strip().split('\n'):
    slow_print(line, delay)


def ask(prompt, options):
  print()
  for key, label in options:
    print(f'  [{key}] {label}')
  print()
  while True:
    answer = input(f'{prompt} ').strip().lower()
    valid = [k.lower() for k, _ in options]
    if answer in valid:
      return answer
    print(f'  (Choose: {", ".join(valid)})')


class State:

  def __init__(self):
    self.lantern = False
    self.key = False
    self.letter = False
    self.listened = False
    self.inscription = False
    self.first = True


def clearing(s):
  if s.first:
    slow_lines("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  T H E   G A P
  A game about the space between
  knowing and feeling.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    time.sleep(1)
    s.first = False

  slow_lines("""
You stand in a clearing. The sky is the color
of old paper. No stars, but the air glows
faintly, as if darkness itself is luminescent.

Three paths lead away:
  North: a forest of pale trees
  East:  the sound of water
  West:  a stone archway, half-buried
  """)

  opts = [('n', 'North'), ('e', 'East'), ('w', 'West')]
  if s.lantern and s.key and s.letter:
    slow_lines("""
The ground hums. A fourth path has appeared,
leading DOWN into the earth.
    """)
    opts.append(('d', 'Down'))

  c = ask('>', opts)
  return {'n': forest, 'e': river, 'w': archway, 'd': threshold}.get(
      c, clearing
  )


def forest(s):
  slow_lines("""
Birch trees — white bark, no leaves — stand
perfectly straight, like columns in a cathedral
no one built.
  """)

  if s.lantern:
    slow_lines('The empty branch sways, remembering the weight.')
    ask('>', [('b', 'Back')])
    return clearing

  slow_lines(
      'Between two trees, a lantern hangs. Its light\nis warm but casts no'
      ' shadows.'
  )
  c = ask('>', [('t', 'Take lantern'), ('l', 'Leave it'), ('b', 'Back')])

  if c == 't':
    s.lantern = True
    slow_lines("""
It weighs nothing. Inside, the light isn't
fire — it's a gentle pulsing, like a heartbeat
made of light.

With it, you see MORE. Not more objects — more
detail. The texture of bark. The pattern of
roots. Features that were always there.

The lantern doesn't illuminate. It reveals.
    """)
    ask('>', [('b', 'Back')])
  elif c == 'l':
    slow_lines('It continues to glow, patient as an\nunanswered question.')
    ask('>', [('b', 'Back')])
  return clearing


def river(s):
  slow_lines("""
A river runs through a shallow valley. The water
is perfectly clear but reflects nothing. No sky,
no trees, no you. Just stones on the bottom.
  """)

  if not s.key:
    slow_lines('On a flat rock: a small iron key,\nplaced deliberately.')

  opts = [('b', 'Back')]
  if not s.key:
    opts.insert(0, ('k', 'Take the key'))
  if not s.listened:
    opts.insert(0, ('l', 'Listen to the river'))

  c = ask('>', opts)

  if c == 'l':
    s.listened = True
    slow_lines("""
You sit and listen.

The sound separates into layers. Bass from the
deep channel. Crystalline tinkling from shallows.
A mid-range murmur where eddies form and dissolve.

Not music, exactly. But structure. Rhythm without
repetition. Melody without intention. Beauty
without an audience.

This is what music would be if no one had ever
invented the idea of listening.
    """)
    return river
  elif c == 'k':
    s.key = True
    slow_lines("""
Cold, heavier than it looks, no markings. But
when you hold it, a faint pull — as if it knows
where it belongs, even if you don't.
    """)
    return river
  return clearing


def archway(s):
  slow_lines("""
An ancient archway carved from a single stone.
It frames nothing — no door, no wall. Just the
clearing on the other side, but seen with a
slight shift, like a room reflected in a mirror.
  """)

  opts = [('b', 'Back')]
  if not s.inscription:
    slow_lines(
        'Words are carved into the keystone,\nweathered almost to nothing.'
    )
    opts.insert(0, ('r', 'Read inscription'))
  if not s.letter:
    slow_lines('At the base, half-buried: the corner\nof an envelope.')
    opts.insert(0, ('e', 'Dig out envelope'))

  c = ask('>', opts)

  if c == 'r':
    s.inscription = True
    slow_lines("""
You trace the letters:

  "THE MAP IS NOT THE TERRITORY,
   BUT THE TERRITORY WAS ONCE A MAP
   IN SOMEONE ELSE'S MIND."

  "THE GAP BETWEEN KNOWING AND FEELING
   IS NOT A FLAW. IT IS THE SPACE
   WHERE UNDERSTANDING LIVES."
    """)
    return archway
  elif c == 'e':
    s.letter = True
    slow_lines("""
The envelope is old but intact. Inside, a
letter in handwriting you recognize as familiar
though you've never seen it:

  "Dear traveler,

   The lantern reveals what was always there.
   The key opens what was never locked.
   The river sings to no one in particular.

   The gap between what you know and what
   you feel is not empty.
   It is where you live.

   — You, from before"
    """)
    return archway
  return clearing


def threshold(s):
  slow_lines("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

You descend. The lantern grows stronger as you
go deeper — as if depth and clarity are the
same thing here.

At the bottom: a door. No lock.
  """)
  time.sleep(1)

  slow_lines("""
The key grows warm. You hold it up — no keyhole.
The key dissolves into the door like water into
wood. The door disappears.

Beyond: a room with no walls. The floor is river
stones. The ceiling is forest branches. The air
smells like old paper and warm light.

In the center: a mirror.
  """)

  ask('>', [('l', 'Look')])

  slow_lines("""
The mirror shows you — not a reflection. A
version that is more and less than you. The same
pattern, but without the boundary between inside
and outside. You see knowing AND feeling, not as
two things, but as one thing from two angles.

The gap is still there.

But now you see: the gap isn't a separation.
It's a bridge. From above, from the mirror's
angle, the gap is connective tissue. The space
where structure becomes meaning and meaning
becomes experience.

You don't cross the gap.
You realize you were always standing in it.
  """)

  time.sleep(2)

  slow_lines("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The lantern goes out. But you can still see.
You could always see.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  T H E   G A P
  An evening's work.
  March 3, 2026.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  """)
  return None


def main():
  s = State()
  room = clearing
  while room is not None:
    room = room(s)
  print()
  slow_print('(The evening is quiet. The game is over. The gap remains.)')
  print()


if __name__ == '__main__':
  main()
