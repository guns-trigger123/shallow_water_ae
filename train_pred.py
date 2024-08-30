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
    num_workers = config["data_params"]["num_workers"]
    train_batch_size = config["data_params"]["train_batch_size"]
    conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                  if re.search(r"\.npy$", i)]
    minmax_data = np.load("data/minmax/minmax_data.npy")
    dataset = ShallowWaterPredictDataset(data_path, conditions, minmax_data, cae)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=train_batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             )
    return dataset, dataloader


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
                                             num_workers=3,
                                             )
    return dataset, dataloader


if __name__ == '__main__':
    device = torch.device('cuda')
    config = yaml.load(open("ae.yaml", "r"), Loader=yaml.FullLoader)
    # cae = init_cae_model(config, "./saved_models/ae_999_400.pt")
    cae_lstm = init_cae_lstm_model(config)
    cae_lstm = cae_lstm.to(device)
    # dataset, dataloader = init_pred_data(config, "train", cae)
    dataset, dataloader = init_latent_pred_data(config, "train")
    criterion = torch.nn.MSELoss()
    opt = optim.Adam(cae_lstm.parameters(), lr=config["exp_params"]["LR"],
                     weight_decay=config["exp_params"]["weight_decay"])
    step_schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=config["exp_params"]["T_0"],
                                                                   T_mult=config["exp_params"]["T_mult"])

    cae_lstm.train()
    for epoch in range(config["trainer_params"]["max_epochs"]):
        t1 = time.time()
        for iter, batch in enumerate(dataloader):
            batch_input = batch["input"].to(device)
            batch_target = batch["target"].to(device)
            R, Hp = batch["R"].reshape(-1, 1, 1) / 40.0, batch["Hp"].reshape(-1, 1, 1) / 20.0
            R, Hp = R.repeat(1, 3, 1), Hp.repeat(1, 3, 1)
            ref_reg = torch.cat([R, Hp], dim=2).to(device)
            batch_input = torch.cat([batch_input, ref_reg], dim=2)

            results = cae_lstm(batch_input)
            loss = criterion(results, batch_target)

            loss.backward()
            opt.step()
            step_schedule.step()
            cae_lstm.zero_grad()

            if (iter + 1) % 390 == 0:
                print(f"epoch: {epoch} iter: {iter} " +
                      f"loss: {loss}")

            if (iter + 1) % 390 == 0:
                save_path = os.path.join('./saved_models/', f'lstm_{epoch}_{iter + 1}.pt')
                torch.save(cae_lstm.state_dict(), save_path)
        t2 = time.time()
        print(f"epoch: {epoch} time: {t2 - t1}s ")
