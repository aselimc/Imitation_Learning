import sys
import os
import torch
import pickle
import gzip
from utils import *
from typing import AnyStr
from typing import Tuple
from tensorboard_evaluation import *
from torch.utils.data import DataLoader, TensorDataset
from agent.bc_agent import BCAgent
import numpy as np

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_data(dataset_dir: AnyStr = "./data", frac: int = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    To read the data from zip file
    Args:
        dataset_dir: string that shows data location
        frac: amount of split between validation and training dataset

    Returns:
    Training and Validation data
    """
    data_file = os.path.join(dataset_dir, "data.pkl.gzip")
    data = pickle.load(gzip.open(data_file, "rb"))
    x, y = np.array(data["state"], dtype=np.float), np.array(data["action"], dtype=np.float)
    length = x.shape[0]
    x_train, y_train = x[:int((1 - frac) * length)], y[:int((1 - frac) * length)]
    x_valid, y_valid = x[int((1 - frac) * length):], y[int((1 - frac) * length):]
    return x_train, y_train, x_valid, y_valid


def preprocessing(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, save: bool = False
                  , history_length: int = 1) -> Tuple[torch.TensorDataset, torch.TensorDataset]:
    x_train = image_processing(x_train)
    x_valid = image_processing(x_valid)
    y_train = np.array([action_to_id(action) for action in y_train])
    y_valid = np.array([action_to_id(action) for action in y_valid])
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weights = 1. / class_sample_count
    samples_weight = torch.from_numpy(np.array([weights[t] for t in y_train])).to(device)
    x_train, y_train = history_stack(x_train, y_train, history_length)
    x_valid, y_valid = history_stack(x_valid, y_valid, history_length)
    training_dataset = TensorDataset(torch.from_numpy(x_train).to(device), torch.from_numpy(y_train).to(device))
    validation_dataset = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid))
    if save:
        torch.save(training_dataset, ".data/training")
        training_dataset(validation_dataset, ".data/validation")
    return training_dataset, validation_dataset, samples_weight


def train_model(training_dataset: TensorDataset, validation_dataset: TensorDataset, samples_weight: torch.Tensor,
                batch_size: int, n_minibatches: int, lr: float, run_name: AnyStr, model_dir: AnyStr = "./models",
                tensorboard_dir="./tensorboard",
                history_length: int = 1):
    torch.backends.cudnn.benchmark

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    agent = BCAgent(history_length, lr)
    tensorboard_eval = Evaluation(tensorboard_dir, run_name, ["val_acc", "train_acc", "train_loss"])

    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight))
    train_loader = DataLoader(training_dataset, sampler=sampler, batch_size=batch_size, pin_memory=True,
                              shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size * 12, pin_memory=True, shuffle=True)
    t_loss, t_acc, v_acc = 0, 0, 0
    for i in range(n_minibatches):
        for batch, labels in train_loader:
            batch, labels = batch.to(device), labels.to(device)
