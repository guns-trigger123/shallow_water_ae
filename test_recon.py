import re
import os
import torch
import yaml
import torch.optim as optim
from shallow_water_dataset import ShallowWaterReconstructDataset
from vanilla_vae import VanillaVAE
from matplotlib import pyplot as plt
import numpy as np


def init_model(config: dict):
    in_channels = config["model_params"]["in_channels"]
    latent_dim = config["model_params"]["latent_dim"]
    return VanillaVAE(in_channels, latent_dim)


def init_vae_data(config: dict):
    data_path = config["data_params"]["data_path"]
    num_workers = config["data_params"]["num_workers"]
    train_batch_size = config["data_params"]["train_batch_size"]
    # conditions = [tuple(re.findall(r"\d+", i)) for i in os.listdir(data_path)
    #               if re.search(r"\.npy$", i)]
    conditions = [(10, 8)]
    dataset = ShallowWaterReconstructDataset(data_path, conditions)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=train_batch_size,
                                             shuffle=True,
                                             num_workers=0,
                                             )
    return dataset, dataloader


if __name__ == '__main__':
    device = torch.device('cuda')
    config = yaml.load(open("./vae.yaml", "r"), Loader=yaml.FullLoader)
    dataset, dataloader = init_vae_data(config)
    model = init_model(config)
    model = model.to(device)
    model.load_state_dict(torch.load("./saved_models/vae_27_10.pt"))

    model.eval()
    for epoch in range(1):
        for iter, batch in enumerate(dataloader):
            batch_input = batch["input"].to(device)
            results = model.forward(batch_input)

            recon = results[0][0].detach().cpu().numpy()
            real = results[1][0].cpu().numpy()
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Titles for each channel
            titles = ['u', 'v', 'h']

            # Plot each channel in a separate subplot
            for i in range(3):
                im_recon = axes[0][i].imshow(recon[i])
                im_real = axes[1][i].imshow(real[i])
                axes[0][i].set_title(titles[i])
                axes[0][i].axis('off')  # Hide the axes
                axes[1][i].set_title(titles[i])
                axes[1][i].axis('off')  # Hide the axes
                fig.colorbar(im_recon, ax=axes[0][i], orientation='vertical')
                fig.colorbar(im_real, ax=axes[1][i], orientation='vertical')

            # Display the plot
            plt.show()
            break

