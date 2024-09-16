import os
import torch
import yaml
import time
import numpy as np
import torch.optim as optim
from torch import nn
from utils import init_model, init_recon_data


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
    device = torch.device('cuda:0')
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    train_dataset, train_dataloader = init_recon_data(config, "train")
    val_dataset, val_dataloader = init_recon_data(config, "val")
    model = init_model(config)
    model = model.to(device)
    recon_mlp = FullyConnectNetwork(model.fc1.out_features, 2)
    recon_mlp = recon_mlp.to(device)
    opt = optim.Adam(model.parameters(), lr=config["exp_params"]["LR"], weight_decay=config["exp_params"]["weight_decay"])
    opt_recon_mlp = optim.Adam(recon_mlp.parameters(), lr=config["exp_params"]["recon_MLP_LR"], weight_decay=config["exp_params"]["weight_decay"])
    sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=config["exp_params"]["T_0"], T_mult=config["exp_params"]["T_mult"])
    sch_recon_mlp = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_recon_mlp, T_0=config["exp_params"]["T_0"], T_mult=config["exp_params"]["T_mult"])
    criterion = torch.nn.L1Loss()

    MAX_ITER = len(train_dataloader)
    SAVED_DIRECTORY = f"./saved_models/latent{config['cae']['latent_dim']}/{config['logging_params']['recon_model_save_type']}_{config['logging_params']['recon_model_number']}/"
    SAVED_PREFIX = config["logging_params"]["recon_model_save_type"]
    with open(SAVED_DIRECTORY + 'train_recon.yaml', 'w') as f:
        yaml.safe_dump(config, f)

    model.train()
    recon_mlp.train()
    best_epoch = -1
    min_err = np.array([100, 100, 100], dtype=np.float32)
    for epoch in range(config["trainer_params"]["max_epochs"]):
        t1 = time.time()
        for iter, batch in enumerate(train_dataloader):
            batch_input = batch["input"].to(device)
            R, Hp = batch["R"].reshape(-1, 1) / 40.0, batch["Hp"].reshape(-1, 1) / 20.0
            ref_reg = torch.cat([R, Hp], dim=1).float().to(device)

            results, latent = model(batch_input), model.encoder(batch_input)
            latent_reg = recon_mlp(latent)
            loss1 = criterion(results, batch_input)
            loss2 = criterion(latent_reg, ref_reg)  # conditional

            loss2_weight = config["trainer_params"]["recon_MLP_weight"]
            if config["trainer_params"]["recon_MLP_weight_mode"] == "gradual":
                loss2_weight = loss2_weight * epoch / config["trainer_params"]["max_epochs"]
            loss = loss1 + loss2_weight * loss2

            loss.backward()
            opt.step()
            opt_recon_mlp.step()
            sch.step()
            sch_recon_mlp.step()
            model.zero_grad()
            recon_mlp.zero_grad()

            if (iter + 1) % MAX_ITER == 0:
                print(f"epoch: {epoch} iter: {iter} " +
                      f"loss: {loss} loss1:{loss1} loss2:{loss2} loss2 weight:{loss2_weight}")

            if (iter + 1) % MAX_ITER == 0:
                save_path = os.path.join(SAVED_DIRECTORY, SAVED_PREFIX + f'_{epoch}_{iter + 1}.pt')
                torch.save(model.state_dict(), save_path)

            if (iter + 1) % MAX_ITER == 0:
                checkpoint_dir = SAVED_DIRECTORY
                CHECKPOINT_NAME = "checkpoint_" + SAVED_PREFIX + ".pt"
                checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_NAME)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'model2_state_dict': recon_mlp.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'optimizer2_state_dict': opt_recon_mlp.state_dict(),
                    'scheduler_state_dict': sch.state_dict(),
                    'scheduler2_state_dict': sch_recon_mlp.state_dict(),
                }, checkpoint_path)
        t2 = time.time()
        print(f"epoch: {epoch} train time: {t2 - t1}s ")

        for iter, batch in enumerate(val_dataloader):
            batch_input = batch["input"].to(device)
            results = model(batch_input)
            batch_err = torch.abs(batch_input - results)
            rela_batch_err = batch_err / (1 + batch_input)

            uvh_mean_err = torch.mean(rela_batch_err, dim=(0, 2, 3))
            uvh_mean_err = uvh_mean_err.cpu().detach().numpy()
            print(f"epoch: {epoch} val relative uvh_mean_err {uvh_mean_err}")

            if np.sum(uvh_mean_err) < np.sum(min_err):
                best_epoch = epoch
                min_err = uvh_mean_err
                save_path = os.path.join(SAVED_DIRECTORY, SAVED_PREFIX + '_best.pt')
                torch.save(model.state_dict(), save_path)
                epoch_save_path = os.path.join(SAVED_DIRECTORY, SAVED_PREFIX + '_best_epoch.pt')
                torch.save(torch.tensor([best_epoch]), epoch_save_path)
                print(f"best_epoch: {best_epoch}")
        t3 = time.time()
        print(f"epoch: {epoch} val time: {t3 - t2}s ")
    print(f"best_epoch: {best_epoch}")
