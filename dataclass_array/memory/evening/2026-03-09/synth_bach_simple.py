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

import math
import os
import struct
import wave


def note_to_freq(note_name):
  notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
  octave = int(note_name[-1])
  note = note_name[:-1]
  n = notes.index(note) - 9 + (octave - 4) * 12
  return 440.0 * (2.0 ** (n / 12.0))


def generate_note(freq, duration_sec, sample_rate=44100):
  num_samples = int(duration_sec * sample_rate)
  samples = []

  attack_time = 0.05
  decay_time = 0.05
  sustain_level = 0.8
  release_time = 0.1

  attack_samples = int(attack_time * sample_rate)
  decay_samples = int(decay_time * sample_rate)
  release_samples = int(release_time * sample_rate)

  # Simple ADSR
  for i in range(num_samples):
    t = i / sample_rate
    phase = 2 * math.pi * freq * t

    # Sawtooth approx
    saw = 2 * (t * freq - math.floor(0.5 + t * freq))
    sine = math.sin(phase)
    sub = math.sin(phase / 2)

    val = 0.5 * saw + 0.3 * sine + 0.2 * sub

    # Envelope calculation
    if i < attack_samples:
      env = i / attack_samples
    elif i < attack_samples + decay_samples:
      env = 1.0 - (1.0 - sustain_level) * ((i - attack_samples) / decay_samples)
    elif i > num_samples - release_samples:
      env = sustain_level * (num_samples - i) / release_samples
    else:
      env = sustain_level

    samples.append(val * env)
  return samples


def write_pgm(filename, data, width, height):
  # Normalize data to 0-255
  max_val = max(data) if data else 1
  min_val = min(data) if data else 0
  rng = max(1e-6, max_val - min_val)

  with open(filename, 'wb') as f:
    f.write(f'P5\n{width} {height}\n255\n'.encode('ascii'))
    # Write rows
    for row in range(height):
      # Upside down, so freq 0 is at bottom
      y_idx = height - 1 - row
      row_data = bytearray()
      for col in range(width):
        val = data[y_idx * width + col]
        norm = int(255 * (val - min_val) / rng)
        row_data.append(norm)
      f.write(row_data)


def main():
  note_duration = 0.25
  sample_rate = 44100

  phrase1 = ['G2', 'D3', 'B3', 'A3', 'B3', 'D3']
  phrase2 = ['C3', 'E3', 'C4', 'B3', 'C4', 'E3']
  phrase3 = ['C3', 'F#3', 'C4', 'A3', 'C4', 'F#3']
  phrase4 = ['B2', 'G3', 'B3', 'A3', 'B3', 'G3']

  score = phrase1 * 2 + phrase2 * 2 + phrase3 * 2 + phrase4 * 2

  full_audio = []
  freq_data = []  # Simple time-frequency pairs for visual

  for note in score:
    freq = note_to_freq(note)
    samples = generate_note(freq, note_duration, sample_rate)
    full_audio.extend(samples)
    freq_data.extend([freq] * len(samples))

  # Generate WAV
  out_dir = '/google/src/cloud/epot/evening_03_09/configs/users/epot/_agents/artifacts/evening/2026-03-09'
  wav_path = os.path.join(out_dir, 'bach_prelude.wav')

  max_amp = max(abs(x) for x in full_audio) or 1

  with wave.open(wav_path, 'w') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    for val in full_audio:
      sample = int(32767 * (val / max_amp))
      wav_file.writeframes(struct.pack('<h', sample))
  print(f'Saved audio to {wav_path}')

  # Generate "Spectrogram" PGM image (fake spectrogram since we don't have numpy)
  img_w = 400
  img_h = 200
  pgm_data = [0] * (img_w * img_h)

  # Max freq in the score is C4 ~261Hz. Let's cap at 500Hz for visual
  max_f = 500.0

  total_samples = len(full_audio)
  for col in range(img_w):
    sample_idx = int((col / img_w) * total_samples)
    if sample_idx >= total_samples:
      break
    freq = freq_data[sample_idx]
    y_bin = int((freq / max_f) * img_h)
    if 0 <= y_bin < img_h:
      # draw a thick line for the fundamental
      for dy in range(-2, 3):
        if 0 <= y_bin + dy < img_h:
          pgm_data[(y_bin + dy) * img_w + col] = 255 - abs(dy) * 50

      # draw a harmonic
      h_bin = int(((freq * 2) / max_f) * img_h)
      if 0 <= h_bin < img_h:
        pgm_data[h_bin * img_w + col] = 100

      h_bin2 = int(((freq * 3) / max_f) * img_h)
      if 0 <= h_bin2 < img_h:
        pgm_data[h_bin2 * img_w + col] = 50

  pgm_path = os.path.join(out_dir, 'bach_spectrogram.pgm')
  write_pgm(pgm_path, pgm_data, img_w, img_h)
  print(f'Saved spectrogram to {pgm_path}')


if __name__ == '__main__':
  main()
