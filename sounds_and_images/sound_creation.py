# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 10:11:14 2024

@author: Ion-1
"""

import numpy as np
import scipy.io.wavfile as wav

sfreq = 48000

def note(duration: int = 10, freq: int = 440, sfreq: int = 48000, fade: tuple = (0, 0))->np.ndarray:
    
    sig = np.sin(2*np.pi*freq*np.linspace(0, duration, duration*sfreq))
    
    if fade[0] > 0:
        sig[:fade[0]*sfreq] = sig[:fade[0]*sfreq]*np.linspace(0, 1, fade[0]*sfreq)
    if fade[1] > 0:
        sig[-fade[1]*sfreq:] = sig[-fade[1]*sfreq:]*np.linspace(1, 0, fade[1]*sfreq)
    
    return sig

sound = np.zeros(shape=(0,))

sound_list = [([440,880], 5), ([880], 5)]

for tones, duration in sound_list:
    tone = np.sum([note(duration, freq, sfreq, (0,0)) for freq in tones], axis=0)
    tone = tone/np.max(abs(tone))
    print(sound,tone)
    sound = np.concatenate((sound, tone))

wav.write("sound.wav", sfreq, sound)