import torch
import yaml
from matplotlib import pyplot as plt
import numpy as np
from utils import init_model, init_recon_data

if __name__ == '__main__':
    device = torch.device('cuda')
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    test_dataset, test_dataloader = init_recon_data(config, "test")
    # save_model_path = "./saved_models/latent128/ae_1/ae_best.pt"
    save_model_path = "./saved_models/latent128/conditional_ae_5/conditional_ae_best.pt"
    model = init_model(config, save_model_path)
    model = model.to(device)

    model.eval()
    for epoch in range(1):
        for iter, batch in enumerate(test_dataloader):
            batch_input = batch["input"].to(device)
            results = model(batch_input)

            batch_recon = results.detach().cpu().numpy()
            batch_real = batch_input.cpu().numpy()
            batch_err = np.abs(batch_real - batch_recon)
            rela_batch_err = batch_err / (1 + batch_real)

            uvh_mean_err = np.mean(rela_batch_err, axis=(0, 2, 3))
            print(f"relative mean error: {uvh_mean_err}")

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
