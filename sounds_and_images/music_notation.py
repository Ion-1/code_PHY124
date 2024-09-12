# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:44:34 2024

@author: Ion-1
"""
import numpy as np
import scipy.io.wavfile as wav
import logging
import dataclasses
from fractions import Fraction as Fr


@dataclasses.dataclass
class Note:
    name: str or Fr
    dots: int
    note: str
    sharp_or_flat: str
    octave: int
    rest: bool = False

    @property
    def freq(self):
        return (
            440
            * Fr(2)
            ** (
                Fr(
                    (self.octave - 4) * 12
                    + {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}[self.note]
                    + {"#": 1, "": 0, "b": -1}[self.sharp_or_flat],
                    12,
                )
            )
            if not self.rest
            else 0
        )

    @property
    def val(self):
        if isinstance(self.name, str):
            self.name = {
                "maxima": Fr(32),
                "longa": Fr(16),
                "double whole": Fr(8),
                "whole": Fr(4),
                "half": Fr(2),
                "quarter": Fr(1),
                "eighth": Fr(1, 2),
                "sixteenth": Fr(1, 4),
                "thirty-second": Fr(1, 8),
                "sixty-fourth": Fr(1, 16),
                "hundred twenty-eighth": Fr(1, 32),
                "two hundred fifty-sixth": Fr(1, 64),
            }[self.name]
        return np.sum([Fr(self.name, 2**i) for i in range(self.dots + 1)])


class SoundGenerator:

    def __init__(self, log: logging.Logger):
        log.warning("Why are you initializing a SoundGenerator?")

    @staticmethod
    def _note(
        duration: int = 10, freq: int = 440, sfreq: int = 48000, fade: tuple = (0, 0)
    ) -> np.ndarray:

        sig = np.sin(
            2
            * np.pi
            * freq
            * np.linspace(0, duration.numerator, int(duration * sfreq))
            / duration.denominator
        )

        if fade[0] > 0:
            sig[: int(fade[0] * sfreq)] = sig[: int(fade[0] * sfreq)] * np.linspace(
                0, 1, int(fade[0] * sfreq)
            )
        if fade[1] > 0:
            sig[int(-fade[1] * sfreq) :] = sig[int(-fade[1] * sfreq) :] * np.linspace(
                1, 0, int(fade[1] * sfreq)
            )

        return sig


class MusicalPiece:

    def __init__(self, name, composer, tempo, sequence: list = []):
        self.name = name
        self.composer = composer
        self.tempo = tempo
        self.duration = Fr(60 * 8, tempo * 3)
        self.sequence = sequence

    def add_note(self, name, dots, note, octave):
        self.sequence.append(
            Note(
                name,
                dots,
                note[0],
                "#" if "#" in note else "b" if "b" in note else "",
                octave,
                True if "rest" in note else False,
            )
        )

    def generate_sound(self, file_name=None, sfreq=48000):
        sound = np.zeros(shape=(0,))

        # sound_list = [((int(note.freq) for note in notes), self.duration*notes[0].val) for notes in self.sequence]
        sound_list = [[[int(notes.freq)], self.duration * notes.val] for notes in self.sequence]

        for tones, duration in sound_list:
            tone = np.sum(
                [SoundGenerator._note(duration, freq, sfreq, (0.005, 0.05)) for freq in tones],
                axis=0,
            )
            if np.max(tone):
                tone = tone / np.max(abs(tone))
            sound = np.concatenate((sound, tone))

        wav.write(
            (
                (file_name if file_name.endswith(".wav") else file_name + ".wav")
                if file_name
                else self.name + ".wav"
            ),
            sfreq,
            sound,
        )


if __name__ == "__main__":
    m = MusicalPiece("Fur Elise", "Ludwig van Beethoven", 158)
    m.add_note("sixteenth", 0, "rest", 5)
    m.add_note("sixteenth", 0, "E", 5)
    m.add_note("sixteenth", 0, "D#", 5)
    m.add_note("sixteenth", 0, "E", 5)
    m.add_note("sixteenth", 0, "D#", 5)
    m.add_note("sixteenth", 0, "E", 5)
    m.add_note("sixteenth", 0, "B", 4)
    m.add_note("sixteenth", 0, "D", 5)
    m.add_note("sixteenth", 0, "C", 5)
    m.add_note("sixteenth", 0, "A", 4)
    m.generate_sound()
