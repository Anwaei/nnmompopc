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

# python .\compare_open.py --from_file_n --file_folder_n pics/results_open_20250201-1133_3 --from_file_m --file_folder_m pics/results_open_20250201-1133_3 --from_file_n_f --file_folder_n_f pics/results_open_20250201-1133_3 --from_file_m_f --file_folder_m_f pics/results_open_20250201-1133_3 --from_file_n_m --file_folder_n_m pics/results_open_20250210-0038_0 --from_file_m_m --file_folder_m_m pics/results_open_20250210-0038_0 --from_file_n_b --file_folder_n_b pics/results_open_20250210-0038_7 --from_file_m_b --file_folder_m_b pics/results_open_20250210-0038_14 --shown