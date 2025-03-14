import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(suppress=True, precision=7)

import os
import soundfile as sf

base_path = '/mnt/kiso-qnap/abe/m1/voice_actor_recognition/data/'

# 1フレームを切り出す
data, fs = sf.read(base_path + 'jvs001_VOICEACTRESS100_001.wav')
# print(f'sampling rate:{fs}')

time = np.arange(0, len(data)/fs, 1/fs)

center = len(data) // 2
cuttime = 0.04
x = data[int(center - (cuttime / 2) * fs) : int(center + (cuttime/2) * fs)]
t = time[int(center - (cuttime / 2) * fs) : int(center + (cuttime/2) * fs)]

plt.plot(t * 1000, x)
plt.xlabel("time [ms]")
plt.ylabel("amplitude")
# plt.savefig("./data/waveform.png")

# 窓関数をかけてフーリエ変換する
# ハミング窓をかける
hamming = np.hamming(len(x))
x = x * hamming

N = 2048 # FFTのサンプル数(2の累乗で高速化)

# ナイキスト周波数以下のデータを取り出す
spectre = np.abs(np.fft.fft(x, N))[:N // 2] # フーリエ変換
fscale = np.fft.fftfreq(N, d = 1.0 / fs)[:N // 2] # フーリエ変換の周波数を取得

plt.plot(fscale, spectre)
plt.xlabel("frequency [Hz]")
plt.ylabel("amplitude spectrum")
# plt.savefig("./data/hamming.png")
plt.close()

n_channel = 20 # メルフィルタバンクのチャネル数
df = fs / N    # 周波数解像度(周波数インデックス1あたりの[Hz]幅)
from mel_filter_bank import mel_filter_bank
filterbank, fcenters = mel_filter_bank(fs, N, n_channel)

# メルフィルタバンクのプロット
for c in np.arange(0, n_channel):
    plt.plot(np.arange(0, N // 2) * df, filterbank[c])

plt.title('Mel Filter Bank')
plt.xlabel('Frequency[Hz]')
# plt.savefig("./data/mel_filter_bank.png")

# 振幅スペクトルにメルフィルタバンクを適用
mspectre = np.dot(spectre, filterbank.T)

# 元の振幅スペクトルとフィルタバンクをかけて圧縮したスペクトルを表示
plt.figure(figsize=(13, 5))

plt.plot(fscale, 10 * np.log10(spectre), label='Original Spectrum')
plt.plot(fcenters, 10 * np.log10(mspectre), "o-", label='Mel Spectrum')
plt.xlabel("frequency[Hz]")
plt.ylabel('Amplitude[dB]')
plt.legend()
# plt.savefig("./data/spectrum.png")

from scipy.fftpack.realtransforms import dct

cepstre = dct(10 * np.log10(mspectre), type=2, norm="ortho", axis=-1)
print(cepstre.shape) # (20,)
print(cepstre) 
#  [13.3190939 24.2771886  6.9022144 -3.5544651 12.9122567  0.5574299
#  -1.8089861  1.496971  -6.083573   0.5448475 -2.8065642 -0.0265315
#   0.8130684  0.1490588 -0.9149896 -1.70858    2.6811971 -0.3079438
#   1.3090417 -1.841087 ]
