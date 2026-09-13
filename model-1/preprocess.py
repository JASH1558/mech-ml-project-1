import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from torchvision import transforms
from config import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_path = "CMaps/train_FD001.txt"
test_path = "CMaps/test_FD001.txt"
test_RUL_path = "CMaps/RUL_FD001.txt"

train_load = np.loadtxt(train_path)
test_load = np.loadtxt(test_path)
test_RUL_load = np.loadtxt(test_RUL_path)



def train_dataset_process(load, window_size):

    data = []
    RUL = []

    number_of_engines = int(load[-1, 0])

    for engine_id in range(1, number_of_engines + 1):

        engine = load[load[:, 0] == engine_id]

        failure_cycle = engine[-1, 1]

        # Remove engine ID
        engine = engine[:, 1:]

        for start in range(0, engine.shape[0] - window_size + 1):

            window = engine[start:start + window_size]

            current_cycle = window[-1, 0]

            rul = failure_cycle - current_cycle

            data.append(window)
            RUL.append(rul)

    return np.array(data), np.array(RUL)



def test_dataset_process(load, window_size, test_RUL_load):

    data = []
    RUL = []

    number_of_engines = int(load[-1, 0])

    for engine_id in range(1, number_of_engines + 1):

        engine = load[load[:, 0] == engine_id]

        last_observed_cycle = engine[-1, 1]

        actual_RUL_at_end = test_RUL_load[engine_id - 1]

        # Remove engine ID
        engine = engine[:, 1:]

        for start in range(0, engine.shape[0] - window_size + 1):

            window = engine[start:start + window_size]

            current_cycle = window[-1, 0]

            rul = (
                last_observed_cycle
                - current_cycle
                + actual_RUL_at_end
            )

            data.append(window)
            RUL.append(rul)

    return np.array(data), np.array(RUL)


def normaliser(dataset,dims):
    mean=dataset.mean(dims)
    std=dataset.std(dims)
    std[std==0]=1

    dataset=(dataset-mean)//std

    return dataset

train_data, train_RUL = train_dataset_process(
    train_load,
    window_size
)

test_data, test_RUL = test_dataset_process(
    test_load,
    window_size,
    test_RUL_load
)


feature_mean = train_data.mean(
    axis=(0, 1),
    keepdims=True
)

feature_std = train_data.std(
    axis=(0, 1),
    keepdims=True
)

feature_std[feature_std == 0] = 1.0


train_data = (
    train_data - feature_mean
) / feature_std


test_data = (
    test_data - feature_mean
) / feature_std

imvar=61.92094574156003

rul_mean = train_RUL.mean()

rul_std = train_RUL.std()

print(rul_std)

if rul_std == 0:
    rul_std = 1.0


train_RUL = (
    train_RUL - rul_mean
) / rul_std


test_RUL= (
    test_RUL - rul_mean
) / rul_std



train_data = torch.tensor(
    train_data,
    dtype=torch.float32
).to(device)

train_RUL = torch.tensor(
    train_RUL,
    dtype=torch.float32
).reshape(-1, 1).to(device)

test_data = torch.tensor(
    test_data,
    dtype=torch.float32
).to(device)

test_RUL = torch.tensor(
    test_RUL,
    dtype=torch.float32
).reshape(-1, 1).to(device)


train_dataset=TensorDataset(train_data,train_RUL)
test_dataset=TensorDataset(test_data,test_RUL)

