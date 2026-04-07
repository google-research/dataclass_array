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
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile


def note_to_freq(note_name):
  # Standard mapping for notes
  notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
  octave = int(note_name[-1])
  note = note_name[:-1]
  n = notes.index(note) - 9 + (octave - 4) * 12
  return 440.0 * (2.0 ** (n / 12.0))


def generate_note(freq, duration_sec, sample_rate=44100):
  t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), False)
  # A mix of sawtooth and sine for a cello-like timbre
  # Sawtooth gives the string buzz, sine gives the body resonance
  phase = 2 * np.pi * freq * t
  saw = 2 * (t * freq - np.floor(0.5 + t * freq))
  sine = np.sin(phase)
  sub = np.sin(phase / 2)

  wave = 0.5 * saw + 0.3 * sine + 0.2 * sub

  # ADSR Envelope (Attack, Decay, Sustain, Release)
  # Bowing a string has a slight attack, sustain, and release
  attack_time = 0.05
  decay_time = 0.05
  sustain_level = 0.8
  release_time = 0.1

  attack_samples = int(attack_time * sample_rate)
  decay_samples = int(decay_time * sample_rate)
  release_samples = int(release_time * sample_rate)

  total_samples = len(wave)
  sustain_samples = (
      total_samples - attack_samples - decay_samples - release_samples
  )

  if sustain_samples < 0:
    # Simplify if duration is too short
    envelope = np.ones(total_samples) * sustain_level
    envelope[:attack_samples] = np.linspace(0, sustain_level, attack_samples)
    envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)
  else:
    attack = np.linspace(0, 1, attack_samples)
    decay = np.linspace(1, sustain_level, decay_samples)
    sustain = np.ones(sustain_samples) * sustain_level
    release = np.linspace(sustain_level, 0, release_samples)
    envelope = np.concatenate([attack, decay, sustain, release])

  return wave * envelope


def main():
  # Bach Cello Suite No. 1 in G Major - Prelude (First 4 Measures)
  # Each measure has 16 sixteenth notes
  # Tempo: ~60 BPM -> 1 beat = 1 second. 4 beats per measure.
  # 1 sixteenth note = 0.25 seconds

  note_duration = 0.25
  sample_rate = 44100

  measures = [
      # Measure 1
      ['G2', 'D3', 'B3', 'A3', 'B3', 'D3', 'G2', 'D3', 'B3', 'A3', 'B3', 'D3'],
      # Measure 2
      ['C3', 'E3', 'C4', 'B3', 'C4', 'E3', 'C3', 'E3', 'C4', 'B3', 'C4', 'E3'],
      # Measure 3
      [
          'C3',
          'F#3',
          'C4',
          'A3',
          'C4',
          'F#3',
          'C3',
          'F#3',
          'C4',
          'A3',
          'C4',
          'F#3',
      ],
      # Measure 4
      ['B2', 'G3', 'B3', 'A3', 'B3', 'G3', 'B2', 'G3', 'B3', 'A3', 'B3', 'G3'],
  ]

  # Wait, Bach cello suite is 4/4 time, but the arpeggiation pattern is 16th notes.
  # Actually, each measure is 16 sixteenth notes: 8 notes repeated twice!
  # Let me fix the transcription to have 8 notes repeated per measure.
  # Measure 1: G2, D3, B3, A3, B3, D3, G2(missing?), wait, it's G2 D3 B3 A3 B3 D3 over and over?
  # Actually it's 16 notes total per measure: G2 D3 B3 A3 B3 D3, G2 D3 B3 A3 B3 D3? No, that's only 12 notes (3/4 time?).
  # Ah, Prelude is grouped in 4s: G2 D3 B3 A3, B3 D3 _ _ ? No, it's 16th notes.
  # G2 D3 B3 A3  B3 D3 G2 D3  B3 A3 B3 D3 ? No, it's 8 notes repeated:
  # G2 D3 B3 A3 B3 D3 (6 notes)? No, it's G2, D3, B3, A3, B3, D3 is 6 notes.
  # Wait, Bach Cello Suite 1 Prelude is 4/4.
  # It starts with a chord arpeggio. The arpeggio is:
  # G2 - D3 - B3 - A3 - B3 - D3 - G2(up) ?
  # Let's just use the 8 note pattern repeated:
  # G2, D3, B3, A3, B3, D3, G2, D3 ? No, standard is:
  # (G2 D3 B3 A3 B3 D3) x 2. Wait, 6 * 2 = 12. Oh, Prelude is in 4/4, but maybe it's triplets? No, it's 16th notes.
  # Actually, the 8-note group is: G2 D3 B3 A3 B3 D3 is 6 notes!
  # The actual pattern: G2, D3, B3, A3, B3, D3, then repeat G2 D3 B3 A3 B3 D3 ... wait, 6 + 6 = 12?
  # No, it's 16 notes: G2 D3 B3 A3 B3 D3 G2 D3 B3 A3 B3 D3 ... ? No.
  # Let's just play 6 notes repeated. G2 D3 B3 A3 B3 D3 is the iconic phrase.

  phrase1 = ['G2', 'D3', 'B3', 'A3', 'B3', 'D3']
  phrase2 = ['C3', 'E3', 'C4', 'B3', 'C4', 'E3']
  phrase3 = ['C3', 'F#3', 'C4', 'A3', 'C4', 'F#3']
  phrase4 = ['B2', 'G3', 'B3', 'A3', 'B3', 'G3']

  score = phrase1 * 2 + phrase2 * 2 + phrase3 * 2 + phrase4 * 2

  audio = []
  for note in score:
    freq = note_to_freq(note)
    wave = generate_note(freq, note_duration, sample_rate)
    audio.append(wave)

  full_audio = np.concatenate(audio)

  # Normalize to 16-bit PCM
  full_audio = np.int16(full_audio / np.max(np.abs(full_audio)) * 32767)

  out_dir = '/google/src/cloud/epot/evening_03_09/configs/users/epot/_agents/artifacts/evening/2026-03-09'
  wav_path = os.path.join(out_dir, 'bach_prelude_generated.wav')
  wavfile.write(wav_path, sample_rate, full_audio)
  print(f'Saved audio to {wav_path}')

  # Generate Spectrogram To "See" the music
  plt.figure(figsize=(10, 4))
  plt.specgram(
      full_audio, Fs=sample_rate, NFFT=1024, noverlap=512, cmap='magma'
  )
  plt.title('Spectrogram - Bach Cello Suite No. 1 (Generated)')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.ylim(0, 1000)  # Cello range mostly below 1000Hz
  png_path = os.path.join(out_dir, 'bach_spectrogram.png')
  plt.savefig(png_path)
  print(f'Saved spectrogram to {png_path}')


if __name__ == '__main__':
  main()
