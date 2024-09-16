import os
import time
import torch
import yaml
import numpy as np
import torch.optim as optim
from utils import init_cae_lstm_model, init_latent_pred_data

if __name__ == '__main__':
    device = torch.device('cuda')
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    cae_lstm = init_cae_lstm_model(config)
    cae_lstm = cae_lstm.to(device)
    conv_ae_name = "conditional_ae_5"
    train_dataset, train_dataloader = init_latent_pred_data(config, "train", conv_ae_name)
    val_dataset, val_dataloader = init_latent_pred_data(config, "val", conv_ae_name)
    criterion = torch.nn.L1Loss()
    opt = optim.Adam(cae_lstm.parameters(), lr=config["exp_params"]["LR"],
                     weight_decay=config["exp_params"]["weight_decay"])
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=config["exp_params"]["T_0"],
                                                         T_mult=config["exp_params"]["T_mult"])

    MAX_ITER = len(train_dataloader)
    SAVED_DIRECTORY = f"./saved_models/latent{config['cae']['latent_dim']}/{config['logging_params']['pred_model_save_type']}_{config['logging_params']['pred_model_number']}/{conv_ae_name}/"
    SAVED_PREFIX = config["logging_params"]["pred_model_save_type"]
    with open(SAVED_DIRECTORY + 'train_pred.yaml', 'w') as f:
        yaml.safe_dump(config, f)

    best_epoch = -1
    min_err = np.array([100, 100, 100], dtype=np.float32)
    for epoch in range(config["trainer_params"]["max_epochs"]):
        t1 = time.time()
        for iter, batch in enumerate(train_dataloader):
            cae_lstm.train()
            batch_input, batch_target = batch["input"].to(device), batch["target"].to(device)
            # conditional
            R, Hp = batch["R"].reshape(-1, 1, 1) / 40.0, batch["Hp"].reshape(-1, 1, 1) / 20.0
            R, Hp = R.repeat(1, 5, 1), Hp.repeat(1, 5, 1)
            ref_reg = torch.cat([R, Hp], dim=2).to(device)
            batch_input = torch.cat([batch_input, ref_reg], dim=2)

            results = cae_lstm(batch_input)
            loss = criterion(results, batch_target)

            loss.backward()
            opt.step()
            sch.step()
            cae_lstm.zero_grad()

            if (iter + 1) % MAX_ITER == 0:
                print(f"epoch: {epoch} iter: {iter} " +
                      f"loss: {loss}")

            if (iter + 1) % MAX_ITER == 0:
                save_path = os.path.join(SAVED_DIRECTORY, SAVED_PREFIX + f'_{epoch}_{iter + 1}.pt')
                torch.save(cae_lstm.state_dict(), save_path)

            if (iter + 1) % MAX_ITER == 0:
                checkpoint_dir = SAVED_DIRECTORY
                CHECKPOINT_NAME = "checkpoint_" + SAVED_PREFIX + ".pt"
                checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_NAME)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': cae_lstm.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scheduler_state_dict': sch.state_dict(),
                }, checkpoint_path)
        t2 = time.time()
        print(f"epoch: {epoch} time: {t2 - t1}s ")
        for iter, batch in enumerate(val_dataloader):
            cae_lstm.eval()
            batch_input, batch_target = batch["input"].to(device), batch["target"].to(device)
            # conditional
            R, Hp = batch["R"].reshape(-1, 1, 1) / 40.0, batch["Hp"].reshape(-1, 1, 1) / 20.0
            R, Hp = R.repeat(1, 5, 1), Hp.repeat(1, 5, 1)
            ref_reg = torch.cat([R, Hp], dim=2).to(device)
            batch_input = torch.cat([batch_input, ref_reg], dim=2)

            results = cae_lstm(batch_input)

            batch_pred, batch_real = results, batch_target
            batch_err = torch.abs(batch_real - batch_pred)
            uvh_mean_err = torch.mean(batch_err, dim=(0, 2))
            uvh_mean_err = uvh_mean_err.cpu().detach().numpy()

            if np.sum(uvh_mean_err) < np.sum(min_err):
                best_epoch = epoch
                min_err = uvh_mean_err
                save_path = os.path.join(SAVED_DIRECTORY, SAVED_PREFIX + '_best.pt')
                torch.save(cae_lstm.state_dict(), save_path)
                epoch_save_path = os.path.join(SAVED_DIRECTORY, SAVED_PREFIX + '_best_epoch.pt')
                torch.save(torch.tensor([best_epoch]), epoch_save_path)
                print(f"best_epoch: {best_epoch}")
        t3 = time.time()
        print(f"epoch: {epoch} val time: {t3 - t2}s ")
    print(f"best_epoch: {best_epoch}")
    pass
