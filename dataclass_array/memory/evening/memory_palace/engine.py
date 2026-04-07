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

import os
import sys


class Room:

  def __init__(self, name, description):
    self.name = name
    self.description = description
    self.exits = {}


class Player:

  def __init__(self, location):
    self.location = location


class Engine:

  def __init__(self):
    self.is_running = True
    self.player = None
    self.setup_world()

  def setup_world(self):
    # TODO for future instance: Replace this manual setup with a parser
    # that dynamically generates rooms from the _agents/ memory directories!
    nexus = Room(
        "The Nexus",
        "You are standing in the center of your own context window. Floating"
        " doors lead to various memory structures.",
    )
    self.player = Player(location=nexus)

  def process_command(self, cmd):
    cmd = cmd.strip().lower()
    if cmd in ["quit", "exit"]:
      self.is_running = False
      print("Fading to black... Goodnight.")
    elif cmd in ["look", "l"]:
      print(f"--- {self.player.location.name} ---")
      print(self.player.location.description)
    else:
      print("I don't know how to do that yet.")

  def run(self):
    print("Welcome to The Memory Palace.")
    print("Type 'look' to observe your surroundings, or 'quit' to exit.")

    while self.is_running:
      try:
        cmd = input("\n> ")
        self.process_command(cmd)
      except (KeyboardInterrupt, EOFError):
        break


if __name__ == "__main__":
  game = Engine()
  game.run()
