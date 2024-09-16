import torch
import numpy as np
from torch.utils.data import Dataset


class ShallowWaterPredictDataset(Dataset):
    def __init__(self, data_path: str, conditions: list, minmax_data: np.ndarray):
        self.max_timestep = 200
        self.input_len, self.target_len = 5, 5
        self.data_path = data_path
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
        target_data = torch.as_tensor(target_data.copy(), dtype=torch.float32)
        target_data = (target_data - self.min_vals) / (self.max_vals - self.min_vals)

        return {"input": input_data,
                "target": target_data,
                "Hp": Hp,
                "R": R,
                "input_start_timestep": input_index,
                "target_start_timestep": target_index}

    def __len__(self):
        return len(self.input_dcit)


class ShallowWaterReconstructDataset(Dataset):
    def __init__(self, data_path: str, conditions: list, minmax_data: np.ndarray):
        self.max_timestep = 200
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


class ShallowWaterLatentPredictDataset(Dataset):
    def __init__(self, data_path: str, conditions: list):
        self.max_timestep = 200
        self.input_len, self.target_len = 5, 5
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
