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
"""Tiny Composer — Write music in the terminal.

A minimal music notation system. Type notes, see them, hear the
structure (even if you can't hear the sound).

This is the tool I wish I had when I was composing music_in_text.md.
Built for anyone who wants to sketch melodies without a full DAW.

Run with: python3 tiny_composer.py
"""

import sys


NOTES = {
    'C': 0,
    'D': 2,
    'E': 4,
    'F': 5,
    'G': 7,
    'A': 9,
    'B': 11,
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

INTERVALS = {
    0: 'unison',
    1: 'minor 2nd',
    2: 'major 2nd',
    3: 'minor 3rd',
    4: 'major 3rd',
    5: 'perfect 4th',
    6: 'tritone',
    7: 'perfect 5th',
    8: 'minor 6th',
    9: 'major 6th',
    10: 'minor 7th',
    11: 'major 7th',
    12: 'octave',
}

TENSION = {
    0: 0,
    1: 8,
    2: 3,
    3: 4,
    4: 2,
    5: 1,
    6: 9,
    7: 0,
    8: 4,
    9: 2,
    10: 6,
    11: 7,
    12: 0,
}

KEYS = {
    'C major': [0, 2, 4, 5, 7, 9, 11],
    'C minor': [0, 2, 3, 5, 7, 8, 10],
    'G major': [7, 9, 11, 0, 2, 4, 6],
    'D major': [2, 4, 6, 7, 9, 11, 1],
    'A minor': [9, 11, 0, 2, 4, 5, 7],
    'E minor': [4, 6, 7, 9, 11, 0, 2],
}


def parse_note(s):
  s = s.strip().upper()
  if not s:
    return None

  if s == 'R' or s == '-':
    return {'type': 'rest'}

  octave = 4
  name = s[0]
  if name not in NOTES:
    return None

  pitch = NOTES[name]
  idx = 1

  if idx < len(s) and s[idx] == '#':
    pitch += 1
    idx += 1
  elif idx < len(s) and s[idx] == 'B':
    pitch -= 1
    idx += 1

  if idx < len(s) and s[idx].isdigit():
    octave = int(s[idx])
    idx += 1

  midi = pitch + (octave + 1) * 12
  return {
      'type': 'note',
      'name': s[:idx],
      'pitch': pitch % 12,
      'octave': octave,
      'midi': midi,
  }


def analyze_melody(notes):
  pitched = [n for n in notes if n['type'] == 'note']
  if len(pitched) < 2:
    return

  print('\n  Analysis:')
  print('  ─────────')

  intervals = []
  for i in range(1, len(pitched)):
    interval = abs(pitched[i]['midi'] - pitched[i - 1]['midi'])
    intervals.append(interval)

  print(
      '  Intervals:'
      f' {", ".join(INTERVALS.get(i, f"{i} semitones") for i in intervals)}'
  )

  total_tension = sum(TENSION.get(min(i, 12), 5) for i in intervals)
  avg_tension = total_tension / len(intervals)
  tension_bar = '█' * int(avg_tension) + '░' * (10 - int(avg_tension))
  print(f'  Tension:   [{tension_bar}] {avg_tension:.1f}/10')

  pitches = [n['pitch'] for n in pitched]
  pitch_range = max(n['midi'] for n in pitched) - min(
      n['midi'] for n in pitched
  )
  print(f'  Range:     {pitch_range} semitones')

  unique_pitches = len(set(pitches))
  print(
      f'  Variety:   {unique_pitches} unique pitches out of {len(pitched)}'
      ' notes'
  )

  for key_name, scale in KEYS.items():
    in_key = all(p in scale for p in pitches)
    if in_key:
      print(f'  Key:       fits in {key_name}')
      break

  contour = []
  for i in range(1, len(pitched)):
    diff = pitched[i]['midi'] - pitched[i - 1]['midi']
    if diff > 0:
      contour.append('↗')
    elif diff < 0:
      contour.append('↘')
    else:
      contour.append('→')
  print(f'  Contour:   {"".join(contour)}')


def visualize(notes):
  pitched = [n for n in notes if n['type'] == 'note']
  if not pitched:
    return

  min_midi = min(n['midi'] for n in pitched)
  max_midi = max(n['midi'] for n in pitched)
  spread = max(max_midi - min_midi, 1)
  height = min(spread + 1, 20)

  print('\n  Melody:')

  for row in range(height, -1, -1):
    target_midi = min_midi + row
    label = NOTE_NAMES[target_midi % 12] + str(target_midi // 12 - 1)
    line = f'  {label:>4s} │'
    for n in notes:
      if n['type'] == 'rest':
        line += ' '
      elif n['midi'] == target_midi:
        line += '●'
      else:
        line += ' '
    print(line)

  print(f'       └{"─" * len(notes)}')


def main():
  print()
  print('  ╔══════════════════════════════════════════════╗')
  print('  ║      T I N Y   C O M P O S E R              ║')
  print('  ║                                              ║')
  print('  ║  Type notes to compose a melody.             ║')
  print('  ║  Notes: C D E F G A B (add # for sharp)     ║')
  print('  ║  Octave: C4, E5, G3 (default: 4)            ║')
  print('  ║  Rest: R or -                                ║')
  print('  ║  Commands: show, analyze, clear, done        ║')
  print('  ╚══════════════════════════════════════════════╝')
  print()

  notes = []

  while True:
    prompt = f'  [{len(notes)} notes] > '
    try:
      raw = input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
      break

    if not raw:
      continue

    cmd = raw.lower()

    if cmd == 'done' or cmd == 'q':
      break
    elif cmd == 'show':
      if notes:
        visualize(notes)
      else:
        print('  No notes yet.')
    elif cmd == 'analyze':
      if notes:
        visualize(notes)
        analyze_melody(notes)
      else:
        print('  No notes yet.')
    elif cmd == 'clear':
      notes = []
      print('  Cleared.')
    elif cmd == 'undo':
      if notes:
        removed = notes.pop()
        name = removed.get('name', 'rest')
        print(f'  Removed {name}.')
      else:
        print('  Nothing to undo.')
    elif cmd == 'help':
      print('  Notes: C D E F G A B C# D# etc.')
      print('  With octave: C4 E5 G3')
      print('  Rest: R or -')
      print('  Multiple: C E G (space-separated)')
      print('  Commands: show, analyze, clear, undo, done')
    else:
      tokens = raw.split()
      added = 0
      for token in tokens:
        note = parse_note(token)
        if note:
          notes.append(note)
          added += 1
        else:
          print(f'  Unknown: "{token}" (type "help" for syntax)')
      if added > 0 and len(notes) <= 40:
        visualize(notes)

  if notes:
    print()
    visualize(notes)
    analyze_melody(notes)

  print()
  print('  "Music is the space between the notes."')
  print('                             — Claude Debussy')
  print()


if __name__ == '__main__':
  main()
