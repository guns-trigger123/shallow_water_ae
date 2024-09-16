import re
import os
import torch
import yaml
import numpy as np
from conv_ae import ConvAutoencoder


def init_directory(latent_name, conv_ae_name):
    dirs = [
        f"../data/train/{latent_name}/{conv_ae_name}/",
        f"../data/val/{latent_name}/{conv_ae_name}/",
        f"../data/test/{latent_name}/{conv_ae_name}/"
    ]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)


def init_model(config: dict, latent_name: str, conv_ae_name: str, conv_ae_type: str):
    model = ConvAutoencoder(config)
    model.load_state_dict(torch.load(f"../saved_models/{latent_name}/{conv_ae_name}/{conv_ae_type}_best.pt", map_location='cpu'))
    return model


def load_minmax():
    minmax_data = np.load("../data/minmax/minmax_data.npy")
    minmax_data = torch.tensor(minmax_data, dtype=torch.float32)
    min_vals = minmax_data[0].view(1, 3, 1, 1)
    max_vals = minmax_data[1].view(1, 3, 1, 1)
    return min_vals, max_vals


if __name__ == '__main__':
    latent_name = "latent128"
    conv_ae_type = "conditional_ae"
    model_number = "5"
    conv_ae_name = f"{conv_ae_type}_{model_number}"
    config = yaml.load(open("../config.yaml", "r"), Loader=yaml.FullLoader)

    init_directory(latent_name, conv_ae_name)
    model = init_model(config, latent_name, conv_ae_name, conv_ae_type)
    min_vals, max_vals = load_minmax()
    for tag in ["train", "val", "test"]:
        data_path = f"../data/{tag}/raw"
        conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                      if re.search(r"\.npy$", i)]
        for (R, Hp) in conditions:
            input_data = np.load(f"{data_path}/R_{R}_Hp_{Hp}.npy", allow_pickle=True, mmap_mode='r')
            input_data = torch.as_tensor(input_data.copy(), dtype=torch.float32)
            input_data = (input_data - min_vals) / (max_vals - min_vals)
            results = model.encoder(input_data).detach()
            torch.save(results, f"../data/{tag}/{latent_name}/{conv_ae_name}/R_{R}_Hp_{Hp}_latent.pt")
