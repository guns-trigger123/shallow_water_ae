import re
import os
import torch
import yaml
import numpy as np
from matplotlib import pyplot as plt
from utils import init_model, init_cae_lstm_model, init_latent_pred_data, init_recon_pred_data


def plot_sample(config, device, cae, cae_lstm, latent_dataloader, ts: int):
    for epoch in range(1):
        for iter, batch in enumerate(latent_dataloader):
            cae_lstm.eval()
            batch_input = batch["input"][ts:ts + 1].to(device)
            batch_target = batch["target"][ts:ts + 1].to(device)
            results = cae_lstm(batch_input).detach().squeeze()
            R, Hp, target_index = int(batch["R"][ts]), int(batch["Hp"][ts]), int(batch["target_start_timestep"][ts])
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
                uvh_mean_err = np.mean(err, axis=(1, 2))
                print(f"timestep {timestep} mean error: {uvh_mean_err}")
                fig, axes = plt.subplots(3, 3, figsize=(12, 12))

                # Titles for each channel
                titles = ['u', 'v', 'h']
                fig.suptitle(f"timestep: {timestep}", fontsize=24)

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


if __name__ == '__main__':
    device = torch.device('cuda')
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    cae_name = "conditional_ae_5"
    # cae_name = "ae_1"
    cae_prefix = "conditional_ae"
    # cae_prefix = "ae"
    pred_name = "lstm_1"
    pred_prefix = "lstm"
    print(f"{cae_name} {pred_name}")
    cae = init_model(config, f"saved_models/latent128/{cae_name}/{cae_prefix}_best.pt")
    cae = cae.to(device)
    cae_lstm = init_cae_lstm_model(config, f"saved_models/latent128/{pred_name}/{cae_name}/{pred_prefix}_best.pt")
    cae_lstm = cae_lstm.to(device)
    print(torch.load(f"saved_models/latent128/{pred_name}/{cae_name}/{pred_prefix}_best_epoch.pt"))
    latent_dataset, latent_dataloader = init_latent_pred_data(config, "test", cae_name=cae_name, shuffle=False)
    recon_dataset, recon_dataloader = init_recon_pred_data(config, "test")

    # plot_sample(config, device, cae, cae_lstm, latent_dataloader,1234)

    for epoch in range(1):
        for iter, (batch_latent, batch_recon) in enumerate(zip(latent_dataloader, recon_dataloader)):
            cae_lstm.eval()
            batch_latent_input = batch_latent["input"].to(device)
            batch_latent_target = batch_latent["target"].to(device)
            # conditional
            # R, Hp = batch_latent["R"].reshape(-1, 1, 1) / 40.0, batch_latent["Hp"].reshape(-1, 1, 1) / 20.0
            # R, Hp = R.repeat(1, 5, 1), Hp.repeat(1, 5, 1)
            # ref_reg = torch.cat([R, Hp], dim=2).to(device)
            # batch_latent_input = torch.cat([batch_latent_input, ref_reg], dim=2)

            latent_results = cae_lstm(batch_latent_input).detach().squeeze()
            batch_size = batch_latent_input.shape[0]
            target_len = batch_latent_target.shape[1]
            latent_dim = batch_latent_target.shape[2]

            cae.eval()
            # batch_recon_input = batch_recon["input"].to(device)
            batch_recon_target = batch_recon["target"].to(device)

            pred_target = cae.decoder(latent_results.reshape(batch_size * target_len, latent_dim)).detach()
            pred_target = pred_target.reshape(batch_size, target_len, 3, 128, 128)

            aaa = pred_target.detach().cpu().numpy()
            bbb = batch_recon_target.cpu().numpy()
            err = np.abs(aaa - bbb)
            rela_err = err / (1 + bbb)
            uvh_mean_err = np.mean(rela_err, axis=(0, 1, 3, 4))
            print(f"relative uvh mean error: {uvh_mean_err}")

            ccc = batch_latent_target.detach().cpu().numpy()
            ddd = latent_results.detach().cpu().numpy()
            err2 = np.abs(ccc - ddd)
            latent_mean_err = np.mean(err2, axis=(0, 1, 2))
            print(f"latent mean error: {latent_mean_err}")

            # break
