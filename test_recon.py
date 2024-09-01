import re
import os
import torch
import yaml
import torch.optim as optim
from shallow_water_dataset import ShallowWaterReconstructDataset
from conv_ae import ConvAutoencoder
from matplotlib import pyplot as plt
import numpy as np


def init_model(config: dict):
    return ConvAutoencoder(config)


def init_recon_data(config: dict, tag: str):
    data_path = config["data_params"][tag + "_data_path"]
    num_workers = config["data_params"][tag + "_num_workers"]
    batch_size = config["data_params"][tag + "_batch_size"]
    conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                  if re.search(r"\.npy$", i)]
    # conditions = [(22, 13)]
    minmax_data = np.load("data/minmax/minmax_data.npy")
    dataset = ShallowWaterReconstructDataset(data_path, conditions, minmax_data)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             )
    return dataset, dataloader


if __name__ == '__main__':
    device = torch.device('cuda')
    config = yaml.load(open("ae.yaml", "r"), Loader=yaml.FullLoader)
    dataset, dataloader = init_recon_data(config, "test")
    model = init_model(config)
    model = model.to(device)
    model.load_state_dict(torch.load("./saved_models/latent512/ae_999_400.pt"))
    # model.load_state_dict(torch.load("./saved_models/baseline/ae_999_400.pt"))
    # model.load_state_dict(torch.load("./saved_models/conditional_ae_999_400.pt"))

    model.eval()
    for epoch in range(1):
        for iter, batch in enumerate(dataloader):
            batch_input = batch["input"].to(device)
            results = model(batch_input)

            batch_recon = results.detach().cpu().numpy()
            batch_real = batch_input.cpu().numpy()
            batch_err = np.abs(batch_real - batch_recon)
            uvh_mean_err = np.mean(batch_err, axis=(0, 2, 3))
            print(f"mean error: {uvh_mean_err}")

            recon = batch_recon[0]
            real = batch_real[0]
            err = batch_err[0]
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
