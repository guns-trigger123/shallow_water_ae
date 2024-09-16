import re
import os
import torch
import numpy as np
from shallow_water_dataset import ShallowWaterReconstructDataset, ShallowWaterLatentPredictDataset, ShallowWaterPredictDataset
from conv_ae import ConvAutoencoder
from lstm import LSTMPredictor


def init_model(config: dict, saved_model_path=None):
    if saved_model_path == None:
        return ConvAutoencoder(config)
    else:
        cae = ConvAutoencoder(config)
        cae.load_state_dict(torch.load(saved_model_path, map_location='cpu'))
        return cae


def init_cae_lstm_model(config: dict, cae_lstm_param_path=None):
    if cae_lstm_param_path == None:
        return LSTMPredictor(config)
    else:
        cae_lstm = LSTMPredictor(config)
        cae_lstm.load_state_dict(torch.load(cae_lstm_param_path, map_location='cpu'))
        return cae_lstm


def init_recon_data(config: dict, tag: str):
    data_path = f"./data/{tag}/raw"
    num_workers = config["data_params"][f"{tag}_num_workers"]
    batch_size = config["data_params"][f"{tag}_batch_size"]
    conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                  if re.search(r"\.npy$", i)]
    minmax_data = np.load("./data/minmax/minmax_data.npy")
    dataset = ShallowWaterReconstructDataset(data_path, conditions, minmax_data)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             )
    return dataset, dataloader


def init_latent_pred_data(config: dict, tag: str, cae_name: str, shuffle=True):
    data_path = f"./data/{tag}/latent{config['cae']['latent_dim']}/{cae_name}"
    num_workers = config["data_params"][tag + "_num_workers"]
    batch_size = config["data_params"][tag + "_batch_size"]
    conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                  if re.search(r"\.pt$", i)]
    if shuffle == False:
        conditions.sort()
    dataset = ShallowWaterLatentPredictDataset(data_path, conditions)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             )
    return dataset, dataloader


def init_recon_pred_data(config: dict, tag: str):
    data_path = f"./data/{tag}/raw"
    num_workers = config["data_params"][tag + "_num_workers"]
    batch_size = config["data_params"][tag + "_batch_size"]
    conditions = [tuple(map(int, re.findall(r"\d+", i))) for i in os.listdir(data_path)
                  if re.search(r"\.npy$", i)]
    conditions.sort()
    minmax_data = np.load("data/minmax/minmax_data.npy")
    dataset = ShallowWaterPredictDataset(data_path, conditions, minmax_data)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             )
    return dataset, dataloader
