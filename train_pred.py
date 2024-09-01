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


def init_latent_pred_data(config: dict, tag: str):
    data_path = config["data_params"][tag + "_data_path"]
    num_workers = config["data_params"][tag + "_num_workers"]
    batch_size = config["data_params"][tag + "_batch_size"]
    conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                  if re.search(r"\.npy$", i)]
    dataset = ShallowWaterLatentPredictDataset(data_path, conditions)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
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
    val_dataset, val_dataloader = init_latent_pred_data(config, "val")
    criterion = torch.nn.MSELoss()
    opt = optim.Adam(cae_lstm.parameters(), lr=config["exp_params"]["LR"],
                     weight_decay=config["exp_params"]["weight_decay"])
    step_schedule = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=config["exp_params"]["T_0"],
                                                                   T_mult=config["exp_params"]["T_mult"])

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

            if (iter + 1) % 390 == 0:
                print(f"epoch: {epoch} iter: {iter} " +
                      f"loss: {loss}")

            if (iter + 1) % 390 == 0:
                save_path = os.path.join('./saved_models/', f'lstm_{epoch}_{iter + 1}.pt')
                # save_path = os.path.join('./saved_models/', f'conditional_lstm_{epoch}_{iter + 1}.pt')
                torch.save(cae_lstm.state_dict(), save_path)

            if (iter + 1) % 390 == 0:
                checkpoint_dir = "./saved_models/"
                CHECKPOINT_NAME = "checkpoint_lstm.pt"
                # CHECKPOINT_NAME = "checkpoint_conditional_lstm.pt"
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
                save_path = os.path.join('./saved_models/', f'lstm_best.pt')
                # save_path = os.path.join('./saved_models/', f'conditional_lstm_best.pt')
                torch.save(cae_lstm.state_dict(), save_path)

                torch.save(torch.tensor([best_epoch]),
                           os.path.join('./saved_models/', f'lstm_best_epoch.pt'))
                # torch.save(torch.tensor([best_epoch]),
                #            os.path.join('./saved_models/', f'conditional_lstm_best_epoch.pt'))

        t3 = time.time()
        print(f"epoch: {epoch} val time: {t3 - t2}s ")
    print(f"best_epoch: {best_epoch}")
    pass
