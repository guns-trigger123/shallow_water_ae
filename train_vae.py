import re
import os
import torch
import yaml
import torch.optim as optim
from shallow_water_dataset import ShallowWaterDataset, ShallowWaterReconstructDataset
from vanilla_vae import VanillaVAE


def init_model(config: dict):
    in_channels = config["model_params"]["in_channels"]
    latent_dim = config["model_params"]["latent_dim"]
    return VanillaVAE(in_channels, latent_dim)


def init_vae_data(config: dict):
    data_path = config["data_params"]["data_path"]
    num_workers = config["data_params"]["num_workers"]
    train_batch_size = config["data_params"]["train_batch_size"]
    # conditions = [tuple(re.findall(r"\d+", i)) for i in os.listdir(data_path)
    #               if re.search(r"\.npy$", i)]
    conditions = [(10, 8)]
    dataset = ShallowWaterReconstructDataset(data_path, conditions)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=train_batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             )
    return dataset, dataloader


if __name__ == '__main__':
    device = torch.device('cuda')
    config = yaml.load(open("./vae.yaml", "r"), Loader=yaml.FullLoader)
    dataset, dataloader = init_vae_data(config)
    model = init_model(config)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=config["exp_params"]["LR"],
                     weight_decay=config["exp_params"]["weight_decay"])
    step_schedule = optim.lr_scheduler.StepLR(step_size=10000, gamma=config["exp_params"]["scheduler_gamma"],
                                              optimizer=opt)

    model.train()
    for epoch in range(config["trainer_params"]["max_epochs"]):
        for iter, batch in enumerate(dataloader):
            batch_input = batch["input"].to(device)
            results = model.forward(batch_input)
            train_loss_dict = model.loss_function(*results,
                                                  M_N=config["exp_params"]["kld_weight"], )
            recon_loss, kld = train_loss_dict["Reconstruction_Loss"], train_loss_dict["KLD"]
            loss = train_loss_dict["loss"]
            loss.backward()
            opt.step()
            step_schedule.step()
            model.zero_grad()

            if (iter + 1) % 10 == 0:
                print(f"pinn epoch: {epoch} iter: {iter} " +
                      f"loss: {loss} Reconstruction_Loss: {recon_loss} KLD: {kld}")

            if (iter + 1) % 10 == 0:
                save_path = os.path.join('./saved_models/', f'vae_{epoch}_{iter + 1}.pt')
                torch.save(model.state_dict(), save_path)

    pass
