import re
import os
import time

import torch
import yaml
import numpy as np
import torch.optim as optim
from shallow_water_dataset import ShallowWaterLatentPredictDataset, ShallowWaterReconstructDataset
from conv_ae import ConvAutoencoder
from lstm import LSTMPredictor
from matplotlib import pyplot as plt


def init_cae_model(config: dict, cae_param_path=None):
    if cae_param_path == None:
        return ConvAutoencoder(config)
    else:
        cae = ConvAutoencoder(config)
        cae.load_state_dict(torch.load(cae_param_path))
        return cae


def init_cae_lstm_model(config: dict, cae_lstm_param_path=None):
    if cae_lstm_param_path == None:
        return LSTMPredictor(config)
    else:
        cae_lstm = LSTMPredictor(config)
        cae_lstm.load_state_dict(torch.load(cae_lstm_param_path))
        return cae_lstm


def init_latent_pred_data(config: dict, tag: str):
    data_path = config["data_params"][tag + "_data_path"]
    num_workers = config["data_params"]["num_workers"]
    train_batch_size = config["data_params"]["train_batch_size"]
    conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                  if re.search(r"\.npy$", i)]
    dataset = ShallowWaterLatentPredictDataset(data_path, conditions)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=train_batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             )
    return dataset, dataloader


def init_recon_data(config: dict, tag: str):
    data_path = config["data_params"][tag + "_data_path"]
    num_workers = config["data_params"]["num_workers"]
    train_batch_size = config["data_params"]["train_batch_size"]
    # conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
    #               if re.search(r"\.npy$", i)]
    conditions = [(22, 13)]
    minmax_data = np.load("data/minmax/minmax_data.npy")
    dataset = ShallowWaterReconstructDataset(data_path, conditions, minmax_data)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=train_batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             )
    return dataset, dataloader


if __name__ == '__main__':
    device = torch.device('cuda')
    config = yaml.load(open("ae.yaml", "r"), Loader=yaml.FullLoader)
    cae = init_cae_model(config, "./saved_models/ae_999_400.pt")
    cae = cae.to(device)
    cae_lstm = init_cae_lstm_model(config, "./saved_models/lstm_999_390.pt")
    cae_lstm = cae_lstm.to(device)
    latent_dataset, latent_dataloader = init_latent_pred_data(config, "test")

    for epoch in range(1):
        for iter, batch in enumerate(latent_dataloader):
            cae_lstm.eval()
            batch_input = batch["input"][0:1].to(device)
            batch_target = batch["target"][0:1].to(device)
            results = cae_lstm(batch_input).detach().squeeze()
            R, Hp, target_index = int(batch["R"][0]), int(batch["Hp"][0]), int(batch["target_start_timestep"][0])
            target_len = batch_target.shape[1]

            real_target = np.load(config["data_params"]["test" + "_data_path"] + f"/R_{R}_Hp_{Hp}.npy",
                                  allow_pickle=True,
                                  mmap_mode='r')[target_index:target_index + target_len]
            minmax_data = np.load("data/minmax/minmax_data.npy")
            # minmax_data = torch.tensor(minmax_data, dtype=torch.float32)
            min_vals = minmax_data[0].reshape(1, 3, 1, 1)
            max_vals = minmax_data[1].reshape(1, 3, 1, 1)
            # input_data = torch.as_tensor(input_data.copy(), dtype=torch.float32)
            real_target = (real_target - min_vals) / (max_vals - min_vals)

            pred_target = cae.decoder(results).detach()

            for timestep in range(target_len):
                recon = pred_target[timestep].cpu().numpy()
                real = real_target[timestep]
                err = np.abs(real - recon)
                fig, axes = plt.subplots(3, 3, figsize=(15, 15))

                # Titles for each channel
                titles = ['u', 'v', 'h']

                # Plot each channel in a separate subplot
                for i in range(3):
                    im_recon = axes[0][i].imshow(recon[i])
                    im_real = axes[1][i].imshow(real[i])
                    im_err = axes[2][i].imshow(err[i])
                    axes[0][i].set_title(titles[i])
                    axes[0][i].axis('off')  # Hide the axes
                    axes[1][i].set_title(titles[i])
                    axes[1][i].axis('off')  # Hide the axes
                    axes[2][i].set_title(titles[i])
                    axes[2][i].axis('off')  # Hide the axes
                    fig.colorbar(im_recon, ax=axes[0][i], orientation='vertical')
                    fig.colorbar(im_real, ax=axes[1][i], orientation='vertical')
                    fig.colorbar(im_err, ax=axes[2][i], orientation='vertical')

                # Display the plot
                plt.show()
            break
