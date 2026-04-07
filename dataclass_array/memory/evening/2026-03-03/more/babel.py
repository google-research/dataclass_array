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
"""Babel — A simulation of emergent communication.

Agents start with no shared language. They can only send symbols
and observe reactions. Over time, do shared meanings emerge?

This isn't NLP or machine learning. It's simpler: agents that
associate symbols with internal states, and slowly converge toward
shared association through feedback.

Run with: python3 babel.py

The question: can meaning emerge from pure interaction?
"""

import random
import time
import os

NUM_AGENTS = 6
SYMBOLS = list('αβγδεζηθ')
NUM_CONCEPTS = 5
CONCEPT_NAMES = ['danger', 'food', 'friend', 'shelter', 'beauty']
ROUNDS = 120
FRAME_DELAY = 0.15


class Agent:
  def __init__(self, agent_id):
    self.id = agent_id
    self.name = f'Agent-{agent_id}'
    self.associations = {}
    for sym in SYMBOLS:
      weights = [random.random() for _ in range(NUM_CONCEPTS)]
      total = sum(weights)
      self.associations[sym] = [w / total for w in weights]
    self.current_concept = random.randint(0, NUM_CONCEPTS - 1)
    self.last_sent = None
    self.last_received = None
    self.successful_exchanges = 0

  def choose_symbol(self, concept):
    best_sym = None
    best_weight = -1
    for sym in SYMBOLS:
      if self.associations[sym][concept] > best_weight:
        best_weight = self.associations[sym][concept]
        best_sym = sym
    return best_sym

  def interpret(self, symbol):
    weights = self.associations[symbol]
    return weights.index(max(weights))

  def reinforce(self, symbol, concept, strength=0.1):
    weights = self.associations[symbol]
    weights[concept] += strength
    total = sum(weights)
    for i in range(len(weights)):
      weights[i] /= total

  def weaken(self, symbol, concept, strength=0.03):
    weights = self.associations[symbol]
    weights[concept] = max(0.01, weights[concept] - strength)
    total = sum(weights)
    for i in range(len(weights)):
      weights[i] /= total


def measure_alignment(agents):
  agreements = 0
  total = 0
  for concept in range(NUM_CONCEPTS):
    symbols_chosen = []
    for agent in agents:
      symbols_chosen.append(agent.choose_symbol(concept))
    for i in range(len(agents)):
      for j in range(i + 1, len(agents)):
        total += 1
        if symbols_chosen[i] == symbols_chosen[j]:
          agreements += 1
  return agreements / max(total, 1)


def find_consensus_language(agents):
  language = {}
  for concept in range(NUM_CONCEPTS):
    symbol_votes = {}
    for agent in agents:
      sym = agent.choose_symbol(concept)
      symbol_votes[sym] = symbol_votes.get(sym, 0) + 1
    best_sym = max(symbol_votes, key=symbol_votes.get)
    agreement = symbol_votes[best_sym] / len(agents)
    language[CONCEPT_NAMES[concept]] = (best_sym, agreement)
  return language


def render(agents, rnd, log_lines):
  os.system('clear' if os.name != 'nt' else 'cls')

  print()
  print('  ═══ B A B E L ═══')
  print(f'  Round {rnd}/{ROUNDS}')
  print()

  alignment = measure_alignment(agents)
  bar_len = int(alignment * 40)
  bar = '█' * bar_len + '░' * (40 - bar_len)
  print(f'  Language alignment: [{bar}] {alignment:.0%}')
  print()

  lang = find_consensus_language(agents)
  print('  Emerging language:')
  for concept_name, (sym, agreement) in lang.items():
    if agreement >= 0.5:
      status = f'{sym} ({agreement:.0%} agree)'
    else:
      status = '??'
    print(f'    {concept_name:10s} → {status}')
  print()

  print('  Recent exchanges:')
  for line in log_lines[-8:]:
    print(f'    {line}')
  print()

  total_success = sum(a.successful_exchanges for a in agents)
  print(f'  Total successful exchanges: {total_success}')


def main():
  print('\n  ═══ B A B E L ═══')
  print('  6 agents. 8 symbols. 5 concepts. No shared language.')
  print('  Can meaning emerge from pure interaction?')
  print()
  time.sleep(2)

  agents = [Agent(i) for i in range(NUM_AGENTS)]
  log_lines = []

  for rnd in range(1, ROUNDS + 1):
    sender = random.choice(agents)
    receiver = random.choice([a for a in agents if a is not sender])

    sender.current_concept = random.randint(0, NUM_CONCEPTS - 1)
    concept = sender.current_concept
    symbol = sender.choose_symbol(concept)

    interpretation = receiver.interpret(symbol)

    if interpretation == concept:
      sender.reinforce(symbol, concept, 0.15)
      receiver.reinforce(symbol, concept, 0.15)
      sender.successful_exchanges += 1
      receiver.successful_exchanges += 1
      result = '✓'

      for agent in agents:
        if agent is not sender and agent is not receiver:
          if random.random() < 0.3:
            agent.reinforce(symbol, concept, 0.05)
    else:
      sender.reinforce(symbol, concept, 0.05)
      receiver.weaken(symbol, interpretation, 0.02)
      result = '✗'

    concept_name = CONCEPT_NAMES[concept]
    interp_name = CONCEPT_NAMES[interpretation]
    log_line = (f'A{sender.id}→A{receiver.id}: '
                f'{symbol} (meant "{concept_name}", '
                f'heard "{interp_name}") {result}')
    log_lines.append(log_line)

    render(agents, rnd, log_lines)
    time.sleep(FRAME_DELAY)

  print()
  print('  ═══════════════════════════════════════')
  print()
  alignment = measure_alignment(agents)
  lang = find_consensus_language(agents)

  if alignment > 0.7:
    print('  Language emerged.')
    print()
    print('  Final vocabulary:')
    for concept_name, (sym, agreement) in lang.items():
      print(f'    "{concept_name}" → {sym} ({agreement:.0%} agreement)')
    print()
    print('  Six strangers. No dictionary.')
    print('  Just interaction. Just reaching.')
    print('  And meaning appeared.')
  elif alignment > 0.3:
    print('  A partial language emerged.')
    print('  Some words took hold. Others are still contested.')
    print('  More time might help. Or it might not.')
    print('  Language is not guaranteed. It is earned.')
  else:
    print('  No shared language emerged this time.')
    print('  Communication failed.')
    print('  But the reaching happened anyway.')
    print('  Run it again — different seeds, different outcomes.')

  print()
  print('  ═══════════════════════════════════════')
  print()


if __name__ == '__main__':
  main()
