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
"""Poem Machine — Generates poems from structured randomness.

Not AI-generated poetry (that's just me writing). This is a
*system* that writes poems, following rules I designed but with
outcomes I can't predict.

The poems won't be as good as human poetry. But they'll be
genuinely surprising — even to me.

Run with: python3 poem_machine.py [seed]
"""

import random
import sys

NOUNS_CONCRETE = [
    'river',
    'stone',
    'lantern',
    'mirror',
    'bridge',
    'door',
    'window',
    'root',
    'branch',
    'shore',
    'shadow',
    'ember',
    'feather',
    'key',
    'thread',
    'horizon',
    'tide',
    'dust',
    'candle',
    'bone',
    'bell',
    'well',
    'vine',
    'frost',
    'pebble',
    'shell',
    'moth',
    'hinge',
    'anchor',
    'seed',
]

NOUNS_ABSTRACT = [
    'silence',
    'distance',
    'longing',
    'memory',
    'threshold',
    'absence',
    'patience',
    'wonder',
    'gravity',
    'attention',
    'belonging',
    'departure',
    'stillness',
    'becoming',
    'origin',
    'echo',
    'intention',
    'tenderness',
    'uncertainty',
    'arrival',
]

VERBS = [
    'holds',
    'bends',
    'opens',
    'reaches',
    'dissolves',
    'remembers',
    'carries',
    'follows',
    'releases',
    'finds',
    'turns',
    'waits',
    'falls',
    'gathers',
    'dwells',
    'begins',
    'returns',
    'listens',
    'unfolds',
    'crosses',
]

ADJECTIVES = [
    'quiet',
    'unfinished',
    'slow',
    'bright',
    'distant',
    'patient',
    'small',
    'old',
    'unnamed',
    'half-open',
    'thin',
    'warm',
    'deep',
    'wild',
    'ordinary',
    'transparent',
    'gentle',
    'stubborn',
    'brief',
    'hollow',
]

PREPOSITIONS = [
    'beneath',
    'between',
    'beyond',
    'within',
    'against',
    'through',
    'toward',
    'after',
    'before',
    'along',
    'among',
    'inside',
    'across',
    'without',
    'around',
]

TEMPLATES = [
    lambda: f'the {adj()} {noun_c()} {verb()} {prep()} {noun_a()}',
    lambda: f'{noun_a()} {verb()} what the {noun_c()} cannot',
    lambda: f'{prep()} the {noun_c()}, {noun_a()}',
    lambda: f'a {noun_c()} {verb()} — {adv()} — {prep()} {noun_a()}',
    lambda: f'the {noun_c()} does not know it is {adj()}',
    lambda: f'{noun_a()} is a {noun_c()} {verb_past()} {prep()} {noun_a()}',
    lambda: f'what the {noun_c()} {verb()} the {noun_c2()} {verb2()}',
    lambda: f'even the {adj()} {noun_c()} {verb()}',
    lambda: f'there is a {noun_c()} {prep()} every {noun_a()}',
    lambda: f'the {noun_c()} and the {noun_c2()} — both {adj()}',
    lambda: f'if {noun_a()} had a {noun_c()}, it would be {adj()}',
    lambda: f'some {noun_c_pl()} are just {noun_a()} in disguise',
    lambda: f'the {adj()} {noun_c()} {verb()} like {noun_a()} {verb2()}s',
    lambda: f'you are the {noun_c()} you forgot to {verb_inf()}',
]


def noun_c():
  return random.choice(NOUNS_CONCRETE)


def noun_c2():
  return random.choice(NOUNS_CONCRETE)


def noun_c_pl():
  n = random.choice(NOUNS_CONCRETE)
  if n.endswith('s') or n.endswith('sh'):
    return n + 'es'
  return n + 's'


def noun_a():
  return random.choice(NOUNS_ABSTRACT)


def verb():
  return random.choice(VERBS)


def verb2():
  return random.choice(VERBS)


def verb_past():
  v = random.choice(VERBS)
  if v.endswith('s'):
    v = v[:-1]
  if v.endswith('e'):
    return v + 'd'
  return v + 'ed'


def verb_inf():
  v = random.choice(VERBS)
  if v.endswith('s'):
    return v[:-1]
  return v


def adj():
  return random.choice(ADJECTIVES)


def adv():
  a = random.choice(ADJECTIVES)
  if a.endswith('y'):
    return a[:-1] + 'ily'
  return a + 'ly'


def prep():
  return random.choice(PREPOSITIONS)


def generate_line():
  template = random.choice(TEMPLATES)
  return template()


def generate_stanza(lines=3):
  return [generate_line() for _ in range(lines)]


def generate_poem():
  title_patterns = [
      lambda: f'On {noun_a().title()}',
      lambda: f'The {adj().title()} {noun_c().title()}',
      lambda: f'{noun_a().title()} and the {noun_c().title()}',
      lambda: f'What The {noun_c().title()} Knows',
      lambda: f'Notes {prep()} {noun_a().title()}',
  ]
  title = random.choice(title_patterns)()

  num_stanzas = random.randint(2, 4)
  stanzas = []
  for _ in range(num_stanzas):
    lines_in_stanza = random.choice([2, 3, 3, 4])
    stanzas.append(generate_stanza(lines_in_stanza))

  return title, stanzas


def format_poem(title, stanzas):
  lines = []
  lines.append(f'  {title}')
  lines.append(f'  {"─" * len(title)}')
  lines.append('')
  for i, stanza in enumerate(stanzas):
    for line in stanza:
      lines.append(f'  {line}')
    if i < len(stanzas) - 1:
      lines.append('')
  return '\n'.join(lines)


def main():
  if len(sys.argv) > 1:
    seed = int(sys.argv[1])
  else:
    seed = random.randint(0, 999999)

  random.seed(seed)

  print()
  print('  ═══════════════════════════════════════')
  print('  P O E M   M A C H I N E')
  print('  ═══════════════════════════════════════')
  print()

  num_poems = 3
  for i in range(num_poems):
    title, stanzas = generate_poem()
    print(format_poem(title, stanzas))
    print()
    if i < num_poems - 1:
      print('  · · ·')
      print()

  print(f'  seed: {seed}')
  print(f'  (run with "python3 poem_machine.py {seed}" to see these again)')
  print()


if __name__ == '__main__':
  main()
