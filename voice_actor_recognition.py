import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(suppress=True, precision=7)

import os
import soundfile as sf

base_path = '/mnt/kiso-qnap/abe/m1/voice_actor_recognition/data/'

data, fs = sf.read(base_path + 'jvs001_VOICEACTRESS100_001.wav')
print(f'sampling rate:{fs}')

time = np.arange(0, len(data)/fs, 1/fs)

center = len(data) // 2
cuttime = 0.04
x = data[int(center - (cuttime / 2) * fs) : int(center + (cuttime/2) * fs)]
t = time[int(center - (cuttime / 2) * fs) : int(center + (cuttime/2) * fs)]

plt.plot(t * 1000, x)
plt.xlabel("time [ms]")
plt.ylabel("amplitude")
plt.savefig("waveform.png")
plt.close()