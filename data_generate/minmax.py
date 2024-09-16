import re
import os
import numpy as np


def minmax():
    data_path = "../data/train/raw"
    conditions = [tuple(re.findall(r"\d+", i)) for i in os.listdir(data_path) if re.search(r"\.npy$", i)]
    max_uvh = np.ones([3]) * (-100)  # 100 is regarded as a big number
    min_uvh = np.ones([3]) * (+100)

    for R, Hp in conditions:
        data = np.load(data_path + f'/R_{R}_Hp_{Hp}.npy', allow_pickle=True, mmap_mode='r')
        max = data.max(axis=(0, 2, 3))
        min = data.min(axis=(0, 2, 3))
        max_uvh = np.maximum(max_uvh, max)
        min_uvh = np.minimum(min_uvh, min)

    max_uvh, min_uvh = max_uvh.reshape(1, -1), min_uvh.reshape(1, -1)
    minmax_uvh = np.concatenate([min_uvh, max_uvh], axis=0)
    return minmax_uvh


if __name__ == '__main__':
    minmax()
