import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data_generate import OptimalDataset

def reload_data(dataset_path):
    dataset = torch.load(dataset_path)
    data_dict = dataset.__dict__
    print("Dataset attributes:")
    for attr, value in dataset.__dict__.items():
        print(f"{attr}: {value}")

if __name__ == "__main__":
    dataset_path = "data/opt_data_02-22-2343.pt"
    reload_data(dataset_path)

