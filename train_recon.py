import re
import os
import torch
import yaml
import numpy as np
import torch.optim as optim
from shallow_water_dataset import ShallowWaterReconstructDataset
from conv_ae import ConvAutoencoder
from torch import nn


def init_model(config: dict):
    return ConvAutoencoder(config)


def init_recon_data(config: dict, tag: str):
    data_path = config["data_params"][tag + "_data_path"]
    num_workers = config["data_params"]["num_workers"]
    train_batch_size = config["data_params"][tag + "_batch_size"]
    conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                  if re.search(r"\.npy$", i)]
    minmax_data = np.load("data/minmax/minmax_data.npy")
    dataset = ShallowWaterReconstructDataset(data_path, conditions, minmax_data)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=train_batch_size,
                                             shuffle=True,
                                             num_workers=3,
                                             )
    return dataset, dataloader


class FullyConnectNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, output_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        for layer in self.fcn:
            x = layer(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda')
    config = yaml.load(open("ae.yaml", "r"), Loader=yaml.FullLoader)
    dataset, dataloader = init_recon_data(config, "train")
    val_dataset, val_dataloader = init_recon_data(config, "val")
    model = init_model(config)
    model = model.to(device)
    model2 = FullyConnectNetwork(model.fc1.out_features, 2)
    model2 = model2.to(device)
    opt = optim.Adam(model.parameters(), lr=config["exp_params"]["LR"],
                     weight_decay=config["exp_params"]["weight_decay"])
    opt.add_param_group({'params': model2.parameters(), 'lr': 0.00005, 'weight_decay': 0.0})
    step_schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=config["exp_params"]["T_0"],
                                                                   T_mult=config["exp_params"]["T_mult"])
    criterion = torch.nn.MSELoss()

    model.train()
    best_epoch = -1
    min_err = np.array([100, 100, 100], dtype=np.float32)
    for epoch in range(config["trainer_params"]["max_epochs"]):
        for iter, batch in enumerate(dataloader):
            batch_input = batch["input"].to(device)
            results = model(batch_input)
            latent = model.encoder(batch_input)
            latent_reg = model2(latent)
            R, Hp = batch["R"].reshape(-1, 1) / 40.0, batch["Hp"].reshape(-1, 1) / 20.0
            ref_reg = torch.cat([R, Hp], dim=1).float().to(device)
            loss1 = criterion(results, batch_input)
            loss2 = criterion(latent_reg, ref_reg)  # conditional
            loss = loss1 + 0.2 * loss2

            loss.backward()
            opt.step()
            step_schedule.step()
            model.zero_grad()

            if (iter + 1) % 400 == 0:
                print(f"epoch: {epoch} iter: {iter} " +
                      f"loss: {loss} loss1:{loss1} loss2:{loss2}")

            if (iter + 1) % 400 == 0:
                # save_path = os.path.join('./saved_models/', f'ae_{epoch}_{iter + 1}.pt')
                save_path = os.path.join('./saved_models/', f'conditional_ae_{epoch}_{iter + 1}.pt')
                torch.save(model.state_dict(), save_path)

        for iter, batch in enumerate(val_dataloader):
            batch_input = batch["input"].to(device)
            results = model(batch_input)

            batch_recon = results.detach().cpu().numpy()
            batch_real = batch_input.cpu().numpy()
            batch_err = np.abs(batch_real - batch_recon)
            uvh_mean_err = np.mean(batch_err, axis=(0, 2, 3))
            if np.sum(uvh_mean_err) < np.sum(min_err):
                best_epoch = epoch
                min_err = uvh_mean_err
                save_path = os.path.join('./saved_models/', f'conditional_ae_best.pt')
                torch.save(model.state_dict(), save_path)
    print(f"best_epoch: {best_epoch}")
    pass
