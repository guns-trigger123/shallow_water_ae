import re
import os
import time

import torch
import yaml
import numpy as np
import torch.optim as optim
from shallow_water_dataset import ShallowWaterPredictDataset, ShallowWaterLatentPredictDataset
from conv_ae import ConvAutoencoder
from lstm import LSTMPredictor


def init_cae_model(config: dict, cae_param_path=None):
    if cae_param_path == None:
        return ConvAutoencoder(config)
    else:
        cae = ConvAutoencoder(config)
        cae.load_state_dict(torch.load(cae_param_path))
        return cae


def init_cae_lstm_model(config: dict):
    return LSTMPredictor(config)


def init_pred_data(config: dict, tag: str, cae: ConvAutoencoder):
    data_path = config["data_params"][tag + "_data_path"]
    num_workers = config["data_params"][tag + "_num_workers"]
    batch_size = config["data_params"][tag + "_batch_size"]
    conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                  if re.search(r"\.npy$", i)]
    minmax_data = np.load("data/minmax/minmax_data.npy")
    dataset = ShallowWaterPredictDataset(data_path, conditions, minmax_data, cae)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             )
    return dataset, dataloader


def init_latent_pred_data(config: dict, tag: str, cae_name: str):
    data_path = config["data_params"][tag + "_data_path"] + "/latent512/" + cae_name
    num_workers = config["data_params"][tag + "_num_workers"]
    batch_size = config["data_params"][tag + "_batch_size"]
    conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                  if re.search(r"\.pt$", i)]
    dataset = ShallowWaterLatentPredictDataset(data_path, conditions)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             )
    return dataset, dataloader


if __name__ == '__main__':
    device = torch.device('cuda:0')
    config = yaml.load(open("ae.yaml", "r"), Loader=yaml.FullLoader)
    cae_lstm = init_cae_lstm_model(config)
    cae_lstm = cae_lstm.to(device)
    cae_name = "conditional_ae_6"
    dataset, dataloader = init_latent_pred_data(config, "train", cae_name)
    val_dataset, val_dataloader = init_latent_pred_data(config, "val", cae_name)
    criterion = torch.nn.MSELoss()
    opt = optim.Adam(cae_lstm.parameters(), lr=config["exp_params"]["LR"],
                     weight_decay=config["exp_params"]["weight_decay"])
    step_schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=config["exp_params"]["T_0"],
                                                                   T_mult=config["exp_params"]["T_mult"])

    MAX_ITER = len(dataloader)
    SAVED_DIRECTORY = config["logging_params"]["pred_model_save_dir"]
    SAVED_PREFIX = config["logging_params"]["pred_model_save_prefix"]
    with open(SAVED_DIRECTORY + 'train_pred.yaml', 'w') as f:
        yaml.safe_dump(config, f)

    best_epoch = -1
    min_err = np.array([100, 100, 100], dtype=np.float32)
    for epoch in range(config["trainer_params"]["max_epochs"]):
        t1 = time.time()
        for iter, batch in enumerate(dataloader):
            cae_lstm.train()
            batch_input = batch["input"].to(device)
            batch_target = batch["target"].to(device)
            # conditional
            # R, Hp = batch["R"].reshape(-1, 1, 1) / 40.0, batch["Hp"].reshape(-1, 1, 1) / 20.0
            # R, Hp = R.repeat(1, 3, 1), Hp.repeat(1, 3, 1)
            # ref_reg = torch.cat([R, Hp], dim=2).to(device)
            # batch_input = torch.cat([batch_input, ref_reg], dim=2)

            results = cae_lstm(batch_input)
            loss = criterion(results, batch_target)

            loss.backward()
            opt.step()
            step_schedule.step()
            cae_lstm.zero_grad()

            if (iter + 1) % MAX_ITER == 0:
                print(f"epoch: {epoch} iter: {iter} " +
                      f"loss: {loss}")

            if (iter + 1) % MAX_ITER == 0:
                save_path = os.path.join(SAVED_DIRECTORY,
                                         SAVED_PREFIX + f'_{epoch}_{iter + 1}.pt')
                torch.save(cae_lstm.state_dict(), save_path)

            if (iter + 1) % MAX_ITER == 0:
                checkpoint_dir = SAVED_DIRECTORY
                CHECKPOINT_NAME = "checkpoint_" + SAVED_PREFIX + ".pt"
                checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_NAME)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': cae_lstm.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': step_schedule.state_dict(),
                }, checkpoint_path)
        t2 = time.time()
        print(f"epoch: {epoch} time: {t2 - t1}s ")
        for iter, batch in enumerate(val_dataloader):
            cae_lstm.eval()
            batch_input = batch["input"].to(device)
            batch_target = batch["target"].to(device)
            # conditional
            # R, Hp = batch["R"].reshape(-1, 1, 1) / 40.0, batch["Hp"].reshape(-1, 1, 1) / 20.0
            # R, Hp = R.repeat(1, 3, 1), Hp.repeat(1, 3, 1)
            # ref_reg = torch.cat([R, Hp], dim=2).to(device)
            # batch_input = torch.cat([batch_input, ref_reg], dim=2)

            results = cae_lstm(batch_input)

            batch_pred = results
            batch_real = batch_target
            batch_err = torch.abs(batch_real - batch_pred)
            uvh_mean_err = torch.mean(batch_err, dim=(0, 2))
            uvh_mean_err = uvh_mean_err.cpu().detach().numpy()

            if np.sum(uvh_mean_err) < np.sum(min_err):
                best_epoch = epoch
                min_err = uvh_mean_err
                save_path = os.path.join(SAVED_DIRECTORY,
                                         SAVED_PREFIX + '_best.pt')
                torch.save(cae_lstm.state_dict(), save_path)

                epoch_save_path = os.path.join(SAVED_DIRECTORY,
                                               SAVED_PREFIX + '_best_epoch.pt')
                torch.save(torch.tensor([best_epoch]), epoch_save_path)
                print(f"best_epoch: {best_epoch}")
        t3 = time.time()
        print(f"epoch: {epoch} val time: {t3 - t2}s ")
    print(f"best_epoch: {best_epoch}")
    pass
