# shallow water dataset
import torch
import numpy as np
from torch.utils.data import Dataset


class ShallowWaterDataset(Dataset):
    '''
    Example:
    dataset = ShallowWaterDataset("./data", [(5, 2)], "videos")
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=16,
                                             shuffle=True,
                                             num_workers=0,
                                             collate_fn=dataset.collate_fn)
    '''

    def __init__(self, data_path: str, conditions: list, split_type: str):
        self.max_timestep = 100
        self.max_imagesize = 128
        self.input_len = 7
        self.target_len = 1
        self.data_path = data_path
        self.conditions = conditions
        self.split_type = split_type  # "videos" or "series"
        if self.split_type == "videos":
            self.input_dcit = [{"R": R, "Hp": Hp, "input_index": i, "target_index": i + self.input_len}
                               for (R, Hp) in conditions for i in range(self.max_timestep - self.input_len)]
        elif self.split_type == "series":
            self.input_dcit = [{"R": R, "Hp": Hp, "input_index": i, "target_index": i + self.input_len,
                                "position": (j, k)}
                               for (R, Hp) in conditions for i in range(self.max_timestep - self.input_len)
                               for j in range(self.max_imagesize) for k in range(self.max_imagesize)]
        else:
            raise ValueError('input split_type="' + split_type + '", not "videos" or "series"')

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
        if self.split_type == "series":
            input_data = input_data[:, :, input_dict["position"][0], input_dict["position"][1]]
            target_data = target_data[:, :, input_dict["position"][0], input_dict["position"][1]]
        return {"input": torch.as_tensor(input_data),
                "target": torch.as_tensor(target_data),
                "Hp": Hp,
                "R": R,
                "input_start_timestep": input_index,
                "target_start_timestep": target_index,
                "position": input_dict.get("position")}

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
        position = [dict.get("position") for dict in batch]

        return {"input": input_data,
                "target": target_data,
                "Hp": Hp,
                "R": R,
                "input_start_timestep": input_index,
                "target_start_timestep": target_index,
                "position": position}
