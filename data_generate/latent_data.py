import re
import os
import torch
import yaml
import numpy as np
import torch.optim as optim
from shallow_water_dataset import ShallowWaterReconstructDataset
from conv_ae import ConvAutoencoder


def init_model(config: dict):
    return ConvAutoencoder(config)


if __name__ == '__main__':
    config = yaml.load(open("../ae.yaml", "r"), Loader=yaml.FullLoader)
    tag = "train"
    data_path = "../" + config["data_params"][tag + "_data_path"]

    conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                  if re.search(r"\.npy$", i)]

    minmax_data = np.load("../data/minmax/minmax_data.npy")
    minmax_data = torch.tensor(minmax_data, dtype=torch.float32)
    min_vals = minmax_data[0].view(1, 3, 1, 1)
    max_vals = minmax_data[1].view(1, 3, 1, 1)

    model = init_model(config)
    model_name = "conditional_ae_3"
    model.load_state_dict(torch.load("../saved_models/latent512/" + model_name + f"/{model_name}_best.pt",
                                     map_location='cpu'))

    for (R, Hp) in conditions:
        input_data = np.load(data_path + f"/R_{R}_Hp_{Hp}.npy", allow_pickle=True, mmap_mode='r')
        input_data = torch.as_tensor(input_data.copy(), dtype=torch.float32)
        input_data = (input_data - min_vals) / (max_vals - min_vals)
        results = model.encoder(input_data).detach()
        torch.save(results, "../data/" + tag + "/latent512/" + model_name + f"/R_{R}_Hp_{Hp}_latent.pt")
