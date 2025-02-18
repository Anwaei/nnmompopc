import numpy as np
import torch
from torch import nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import plot_utils as pu

def read_loss_from_tensorboard(folder):
    event_acc = EventAccumulator(folder)
    event_acc.Reload()
    loss_data = event_acc.Scalars('loss')
    return np.array([e.value for e in loss_data])
    # return [(e.wall_time, e.step, e.value) for e in loss_data]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare net efficiency.')
    parser.add_argument('--folder_vanilla_train_loss', type=str, default='runs/run_02-17-1714/loss_train')
    parser.add_argument('--folder_vanilla_test_loss', type=str, default='runs/run_02-17-1714/loss_test')
    parser.add_argument('--folder_attention_train_loss', type=str, default='runs/run_02-14-1054/loss_train')
    parser.add_argument('--folder_attention_test_loss', type=str, default='runs/run_02-14-1054/loss_test')
    args = parser.parse_args()

    folder_vanilla_train_loss = args.folder_vanilla_train_loss
    folder_vanilla_test_loss = args.folder_vanilla_test_loss
    folder_attention_train_loss = args.folder_attention_train_loss
    folder_attention_test_loss = args.folder_attention_test_loss

    loss_data_vanilla_train = read_loss_from_tensorboard(folder_vanilla_train_loss)
    loss_data_vanilla_test = read_loss_from_tensorboard(folder_vanilla_test_loss)
    loss_data_attention_train = read_loss_from_tensorboard(folder_attention_train_loss)
    loss_data_attention_test = read_loss_from_tensorboard(folder_attention_test_loss)

    loss_data = {
        'vanilla_train': loss_data_vanilla_train,
        'vanilla_test': loss_data_vanilla_test,
        'attention_train': loss_data_attention_train,
        'attention_test': loss_data_attention_test
    }

    pu.plot_compare_net(loss_data=loss_data)

    print(loss_data)

    pass