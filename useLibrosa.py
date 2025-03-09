import librosa
import os
import numpy as np
import pandas as pd

base_path = '/mnt/kiso-qnap/abe/m1/voice_actor_recognition/data/'

X_data = []  # 特徴行列
y_data = []  # 正解情報

for speaker in range(1, 101):
    # print(f'No.{speaker_num}') # 進捗確認

    # 音声データが入っているディレクトリ名
    dir_name = base_path + f'jvs_ver1/jvs{str(speaker).zfill(3)}/parallel100/wav24kHz16bit'
    for file_name in os.listdir(dir_name):
         # 音声ファイルへのパス
        file_path = os.path.join(dir_name, file_name)
        # 音声ファイルを読み込む
        y, sr = librosa.load(file_path)

        mfcc = librosa.feature.mfcc(y, sr) # MFCCを算出
        mfcc = np.average(mfcc, axis=1) # 時間平均を取る
        mfcc = mfcc.flatten()
        mfcc = mfcc.tolist()
        X_data.append(mfcc)
        y_data.append(speaker)

X = pd.DataFrame(X_data, columns=[f'mfcc_{n}' for n in range(0, 20)])
y = pd.DataFrame({'target': y_data})

df = pd.concat([X, y], axis=1)
df.to_csv('mfcc.csv', index=False) # CSVで保存

print(df.shape)
df.head()