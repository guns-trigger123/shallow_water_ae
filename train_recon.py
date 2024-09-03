import re
import os
import torch
import yaml
import time
import numpy as np
import torch.optim as optim
from shallow_water_dataset import ShallowWaterReconstructDataset
from conv_ae import ConvAutoencoder
from torch import nn


def init_model(config: dict, saved_model_path=None):
    if saved_model_path == None:
        return ConvAutoencoder(config)
    else:
        cae = ConvAutoencoder(config)
        cae.load_state_dict(torch.load(saved_model_path))
        return cae


def init_recon_data(config: dict, tag: str):
    data_path = config["data_params"][tag + "_data_path"]
    num_workers = config["data_params"][tag + "_num_workers"]
    batch_size = config["data_params"][tag + "_batch_size"]
    conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                  if re.search(r"\.npy$", i)]
    minmax_data = np.load("data/minmax/minmax_data.npy")
    dataset = ShallowWaterReconstructDataset(data_path, conditions, minmax_data)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             )
    return dataset, dataloader


class FullyConnectNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fcn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim),
            # nn.Sigmoid(),
            nn.ReLU(),
        )

    def forward(self, x):
        for layer in self.fcn:
            x = layer(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:7')
    config = yaml.load(open("ae.yaml", "r"), Loader=yaml.FullLoader)
    dataset, dataloader = init_recon_data(config, "train")
    val_dataset, val_dataloader = init_recon_data(config, "val")
    model = init_model(config)
    # model = init_model(config,"./saved_models/latent512/ae1499_LR_0.001/ae_499_400.pt")
    model = model.to(device)
    model2 = FullyConnectNetwork(model.fc1.out_features, 2)
    model2 = model2.to(device)
    opt = optim.Adam(model.parameters(), lr=config["exp_params"]["LR"],
                     weight_decay=config["exp_params"]["weight_decay"])
    opt2 = optim.Adam(model2.parameters(), lr=config["exp_params"]["recon_MLP_LR"],
                      weight_decay=config["exp_params"]["weight_decay"])
    step_schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=config["exp_params"]["T_0"],
                                                                   T_mult=config["exp_params"]["T_mult"])
    step_schedule2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2, T_0=config["exp_params"]["T_0"],
                                                                    T_mult=config["exp_params"]["T_mult"])
    criterion = torch.nn.MSELoss()

    MAX_ITER = len(dataloader)
    # SAVED_DIRECTORY = './saved_models/'
    SAVED_DIRECTORY = config["logging_params"]["model_save_dir"]
    # SAVED_PREFIX = 'ae'
    SAVED_PREFIX = config["logging_params"]["model_save_prefix"]
    with open(SAVED_DIRECTORY + 'train_recon.yaml', 'w') as f:
        yaml.safe_dump(config, f)

    model.train()
    model2.train()
    best_epoch = -1
    min_err = np.array([100, 100, 100], dtype=np.float32)
    for epoch in range(config["trainer_params"]["max_epochs"]):
        t1 = time.time()
        for iter, batch in enumerate(dataloader):
            batch_input = batch["input"].to(device)
            results = model(batch_input)
            latent = model.encoder(batch_input)
            latent_reg = model2(latent)
            R, Hp = batch["R"].reshape(-1, 1) / 40.0, batch["Hp"].reshape(-1, 1) / 20.0
            ref_reg = torch.cat([R, Hp], dim=1).float().to(device)
            loss1 = criterion(results, batch_input)
            loss2 = criterion(latent_reg, ref_reg)  # conditional

            loss2_weight = config["trainer_params"]["recon_MLP_weight"]
            if config["trainer_params"]["recon_MLP_weight_mode"] == "gradual":
                loss2_weight = loss2_weight * epoch / config["trainer_params"]["max_epochs"]
            loss = loss1 + loss2_weight * loss2

            loss.backward()
            opt.step()
            opt2.step()
            step_schedule.step()
            step_schedule2.step()
            model.zero_grad()
            model2.zero_grad()

            if (iter + 1) % MAX_ITER == 0:
                print(f"epoch: {epoch} iter: {iter} " +
                      f"loss: {loss} loss1:{loss1} loss2:{loss2} loss2 weight:{loss2_weight}")

            if (iter + 1) % MAX_ITER == 0:
                save_path = os.path.join(SAVED_DIRECTORY,
                                         SAVED_PREFIX + f'_{epoch}_{iter + 1}.pt')
                torch.save(model.state_dict(), save_path)

            if (iter + 1) % MAX_ITER == 0:
                checkpoint_dir = SAVED_DIRECTORY
                CHECKPOINT_NAME = "checkpoint_" + SAVED_PREFIX + ".pt"
                checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_NAME)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'model2_state_dict': model2.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'optimizer2_state_dict': opt2.state_dict(),
                    'scheduler_state_dict': step_schedule.state_dict(),
                    'scheduler2_state_dict': step_schedule2.state_dict(),
                }, checkpoint_path)
        t2 = time.time()
        print(f"epoch: {epoch} train time: {t2 - t1}s ")

        for iter, batch in enumerate(val_dataloader):
            batch_input = batch["input"].to(device)
            results = model(batch_input)

            batch_recon = results
            batch_real = batch_input
            batch_err = torch.abs(batch_real - batch_recon)
            uvh_mean_err = torch.mean(batch_err, dim=(0, 2, 3))
            uvh_mean_err = uvh_mean_err.cpu().detach().numpy()

            if np.sum(uvh_mean_err) < np.sum(min_err):
                best_epoch = epoch
                min_err = uvh_mean_err
                save_path = os.path.join(SAVED_DIRECTORY,
                                         SAVED_PREFIX + '_best.pt')
                torch.save(model.state_dict(), save_path)

                epoch_save_path = os.path.join(SAVED_DIRECTORY,
                                               SAVED_PREFIX + '_best_epoch.pt')
                torch.save(torch.tensor([best_epoch]), epoch_save_path)
                print(f"best_epoch: {best_epoch}")
        t3 = time.time()
        print(f"epoch: {epoch} val time: {t3 - t2}s ")
    print(f"best_epoch: {best_epoch}")
    pass
