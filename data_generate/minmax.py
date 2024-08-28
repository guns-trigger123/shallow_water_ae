import re
import os

import torch
import yaml
import numpy as np

def minmax():
    config = yaml.load(open("../" + "ae.yaml", "r"), Loader=yaml.FullLoader)
    data_path = config["data_params"]["train_data_path"]
    conditions = [tuple(re.findall(r"\d+", i)) for i in os.listdir("../" + data_path)
                  if re.search(r"\.npy$", i)]

    max_uvh = np.ones([3]) * (-100)
    min_uvh = np.ones([3]) * (+100)

    for R, Hp in conditions:
        data = np.load(f'../data/R_{R}_Hp_{Hp}.npy', allow_pickle=True, mmap_mode='r')
        max = data.max(axis=(0, 2, 3))
        min = data.min(axis=(0, 2, 3))
        max_uvh = np.maximum(max_uvh, max)
        min_uvh = np.minimum(min_uvh, min)

    max_uvh, min_uvh = max_uvh.reshape(1, -1), min_uvh.reshape(1, -1)
    minmax_uvh = np.concatenate([min_uvh, max_uvh], axis=0)
    return minmax_uvh

if __name__ == '__main__':
    # a = minmax()
    # np.save("./minmax_data", a)
    # b = np.load("./minmax123.npy")
    # print(b)
    pass