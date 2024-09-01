# shallow water dataset
import torch
import numpy as np
from torch.utils.data import Dataset
from conv_ae import ConvAutoencoder


class ShallowWaterDataset(Dataset):
    def __init__(self, data_path: str, conditions: list, minmax_data: np.ndarray):
        self.max_timestep = 200
        self.max_imagesize = 128
        self.input_len = 3
        self.target_len = 3
        self.data_path = data_path
        self.conditions = conditions
        minmax_data = torch.tensor(minmax_data, dtype=torch.float32)
        self.min_vals = minmax_data[0].view(3, 1, 1)
        self.max_vals = minmax_data[1].view(3, 1, 1)
        self.input_dcit = [{"R": R, "Hp": Hp, "input_index": i, "target_index": i + self.input_len, }
                           for (R, Hp) in conditions for i in
                           range(self.max_timestep - self.input_len - self.target_len + 1)]

    def __getitem__(self, item):
        input_dict = self.input_dcit[item]
        R, Hp = input_dict["R"], input_dict["Hp"]
        input_index, target_index = input_dict["input_index"], input_dict["target_index"]
        input_data = np.load(self.data_path + f"/R_{R}_Hp_{Hp}.npy",
                             allow_pickle=True,
                             mmap_mode='r')[input_index:input_index + self.input_len]
        input_data = torch.as_tensor(input_data.copy(), dtype=torch.float32)
        input_data = (input_data - self.min_vals) / (self.max_vals - self.min_vals)

        target_data = np.load(self.data_path + f"/R_{R}_Hp_{Hp}.npy",
                              allow_pickle=True,
                              mmap_mode='r')[target_index:target_index + self.target_len]
        return {"input": torch.as_tensor(input_data.copy(), dtype=torch.float32),
                "target": torch.as_tensor(target_data.copy(), dtype=torch.float32),
                "Hp": Hp,
                "R": R,
                "input_start_timestep": input_index,
                "target_start_timestep": target_index}

    def __len__(self):
        return len(self.input_dcit)

    @staticmethod
    def collate_fn(batch):
        input_data = torch.stack([dict["input"] for dict in batch])
        target_data = torch.stack([dict["target"] for dict in batch])
        Hp = [dict["Hp"] for dict in batch]
        R = [dict["R"] for dict in batch]
        input_index = [dict["input_start_timestep"] for dict in batch]
        target_index = [dict["target_start_timestep"] for dict in batch]

        return {"input": input_data,
                "target": target_data,
                "Hp": Hp,
                "R": R,
                "input_start_timestep": input_index,
                "target_start_timestep": target_index}


class ShallowWaterReconstructDataset(Dataset):
    def __init__(self, data_path: str, conditions: list, minmax_data: np.ndarray):
        self.max_timestep = 200
        self.max_imagesize = 128
        self.data_path = data_path
        self.conditions = conditions
        minmax_data = torch.tensor(minmax_data, dtype=torch.float32)
        self.min_vals = minmax_data[0].view(3, 1, 1)
        self.max_vals = minmax_data[1].view(3, 1, 1)
        self.input_dcit = [{"R": R, "Hp": Hp, "input_index": i, }
                           for (R, Hp) in conditions for i in range(self.max_timestep)]

    def __getitem__(self, item):
        input_dict = self.input_dcit[item]
        R, Hp = input_dict["R"], input_dict["Hp"]
        input_index = input_dict["input_index"]
        input_data = np.load(self.data_path + f"/R_{R}_Hp_{Hp}.npy",
                             allow_pickle=True,
                             mmap_mode='r')[input_index]
        input_data = torch.as_tensor(input_data.copy(), dtype=torch.float32)
        input_data = (input_data - self.min_vals) / (self.max_vals - self.min_vals)

        return {"input": input_data,
                "Hp": Hp,
                "R": R, }

    def __len__(self):
        return len(self.input_dcit)


class ShallowWaterPredictDataset(Dataset):
    def __init__(self, data_path: str, conditions: list, minmax_data: np.ndarray, cae: ConvAutoencoder):
        self.max_timestep = 200
        self.max_imagesize = 128
        self.input_len = 3
        self.target_len = 3
        self.data_path = data_path
        self.conditions = conditions
        minmax_data = torch.tensor(minmax_data, dtype=torch.float32)
        self.min_vals = minmax_data[0].view(1, 3, 1, 1)
        self.max_vals = minmax_data[1].view(1, 3, 1, 1)
        self.input_dcit = [{"R": R, "Hp": Hp, "input_index": i, "target_index": i + self.input_len, }
                           for (R, Hp) in conditions for i in
                           range(self.max_timestep - self.input_len - self.target_len + 1)]
        self.cae = cae

    def __getitem__(self, item):
        input_dict = self.input_dcit[item]
        R, Hp = input_dict["R"], input_dict["Hp"]
        input_index, target_index = input_dict["input_index"], input_dict["target_index"]
        input_data = np.load(self.data_path + f"/R_{R}_Hp_{Hp}.npy",
                             allow_pickle=True,
                             mmap_mode='r')[input_index:input_index + self.input_len]
        target_data = np.load(self.data_path + f"/R_{R}_Hp_{Hp}.npy",
                              allow_pickle=True,
                              mmap_mode='r')[target_index:target_index + self.target_len]
        input_data = torch.as_tensor(input_data.copy(), dtype=torch.float32)
        target_data = torch.as_tensor(target_data.copy(), dtype=torch.float32)
        input_data = (input_data - self.min_vals) / (self.max_vals - self.min_vals)
        target_data = (target_data - self.min_vals) / (self.max_vals - self.min_vals)
        input_latent = self.cae.encoder(input_data).detach()
        target_latent = self.cae.encoder(target_data).detach()
        return {"input": input_latent,
                "target": target_latent,
                "Hp": Hp,
                "R": R,
                "input_start_timestep": input_index,
                "target_start_timestep": target_index, }

    def __len__(self):
        return len(self.input_dcit)


class ShallowWaterLatentPredictDataset(Dataset):
    def __init__(self, data_path: str, conditions: list):
        self.max_timestep = 200
        self.latent_size = 512
        self.input_len = 3
        self.target_len = 3
        self.data_path = data_path
        self.conditions = conditions
        self.input_dcit = [{"R": R, "Hp": Hp, "input_index": i, "target_index": i + self.input_len, }
                           for (R, Hp) in conditions for i in
                           range(self.max_timestep - self.input_len - self.target_len + 1)]

    def __getitem__(self, item):
        input_dict = self.input_dcit[item]
        R, Hp = input_dict["R"], input_dict["Hp"]
        input_index, target_index = input_dict["input_index"], input_dict["target_index"]
        input_data = torch.load(self.data_path +
                                f"/R_{R}_Hp_{Hp}_latent.pt")[input_index:input_index + self.input_len]
        target_data = torch.load(self.data_path +
                                 f"/R_{R}_Hp_{Hp}_latent.pt")[target_index:target_index + self.target_len]
        return {"input": input_data,
                "target": target_data,
                "Hp": Hp,
                "R": R,
                "input_start_timestep": input_index,
                "target_start_timestep": target_index, }

    def __len__(self):
        return len(self.input_dcit)
