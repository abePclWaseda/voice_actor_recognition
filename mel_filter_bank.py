import numpy as np

# メルフィルタバンクをかける
def hz2mel(f):
    """Hzをmelに変換"""
    return 2595 * np.log(f / 700.0 + 1.0)

def mel2hz(m):
    """melをHzに変換"""
    return 700 * (np.exp(m / 2595) - 1.0)  

def mel_filter_bank(fs, N, n_channel):
    """メルフィルタバンクを作成"""
    # ナイキスト周波数[Hz]
    fmax = fs / 2
    # ナイキスト周波数[mel]
    melmax = hz2mel(fmax)
    # 周波数インデックスの最大値
    nmax = N // 2
    # 周波数解像度(周波数インデックス1あたりの[Hz]幅)
    df = fs / N

    # メル尺度における各フィルタの中心周波数を求める
    dmel = melmax / (n_channel + 1)
    melcenters = np.arange(1, n_channel + 1) * dmel

    # 各フィルタの中心周波数を[Hz]に変換
    fcenters = mel2hz(melcenters)
    # 各フィルタの中心周波数を周波数インデックスに変換
    indexcenter = np.round(fcenters / df)

    # 各フィルタの開始位置のインデックス
    indexstart = np.hstack(([0], indexcenter[0:n_channel - 1]))
    # 各フィルタの終了位置のインデックス
    indexstop = np.hstack((indexcenter[1:n_channel], [nmax]))

    filterbank = np.zeros((n_channel, nmax))
    # print(indexstop)

    for c in range(0, n_channel): # c: channel
        # 三角フィルタの左の直線の傾きから点を求める
        increment= 1.0 / (indexcenter[c] - indexstart[c]) # 右上がりの傾き

        for i in range(int(indexstart[c]), int(indexcenter[c])):
            filterbank[c, i] = (i - indexstart[c]) * increment

        # 三角フィルタの右の直線の傾きから点を求める
        decrement = 1.0 / (indexstop[c] - indexcenter[c]) # 右下がりの傾き
        for i in range(int(indexcenter[c]), int(indexstop[c])):
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    return filterbank, fcenters
     