import pickle
import gzip
from utils import *
from typing import AnyStr
from typing import Tuple
from tensorboard_evaluation import *
from torch.utils.data import DataLoader, TensorDataset
from agent.bc_agent import BCAgent

import numpy as np
import os
import torch

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_data(dataset_dir= "./data", frac: int = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    x = np.array(data["state"]).astype("float32")
    y = np.array(data["action"]).astype("float32")
    length = x.shape[0]
    x_train, y_train = x[:int((1 - frac) * length)], y[:int((1 - frac) * length)]
    x_valid, y_valid = x[int((1 - frac) * length):], y[int((1 - frac) * length):]
    return x_train, y_train, x_valid, y_valid


def preprocessing(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, save: bool = False
                  , history_length: int = 1):
    x_train = image_processing(x_train)
    x_valid = image_processing(x_valid)
    y_train = torch.LongTensor([action_to_id(action) for action in y_train]).to(device)
    y_valid = torch.HalfTensor([action_to_id(action) for action in y_valid]).to(device)
    class_sample_count = torch.ShortTensor([len(torch.where(y_train == t)[0]) for t in torch.unique(y_train)])
    weights = 1. / class_sample_count
    del class_sample_count
    samples_weight = torch.HalfTensor([weights[t] for t in y_train]).to(device)
    del weights
    print("History Stacking")
    x_train = history_stack(x_train, history_length)
    x_valid = history_stack(x_valid, history_length)
    training_dataset = TensorDataset(x_train, y_train)
    del x_train
    del y_train
    validation_dataset = TensorDataset(x_valid, y_valid)
    del x_valid
    del y_valid
    if save:
        torch.save(training_dataset, os.path.join("./data", "training_data.pt"))
        torch.save(validation_dataset, os.path.join("./data", "validation_data.pt"))
        torch.save(samples_weight, os.path.join("./data", "weights.pt"))
    return training_dataset, validation_dataset, samples_weight


def train_model(training_dataset: TensorDataset, validation_dataset: TensorDataset, samples_weight: torch.Tensor,
                batch_size: int, epochs: int, lr: float, run_name: AnyStr, model_dir: AnyStr = "./models",
                tensorboard_dir="./tensorboard",
                history_length: int = 1):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    agent = BCAgent(history_length, lr)
    tensorboard_eval = Evaluation(tensorboard_dir, run_name, ["val_acc", "train_acc", "train_loss"])

    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight))
    train_loader = DataLoader(training_dataset, sampler=sampler, batch_size=batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    data_size = float(len(train_loader.dataset))
    iteration_count = round(data_size/batch_size-1)
    t_loss = 0
    t_acc = []
    v_acc = []
    for epoch in range(epochs):
        for i, (batch, labels) in enumerate(train_loader):
            loss = agent.update(batch, labels)
            t_loss += loss
            z = 0
            with torch.no_grad():
                for val_batch, val_labels in validation_loader:
                    if z<4:
                        pred_val = agent.predict(val_batch)
                        v_acc.append(accuracy(pred_val, val_labels))
                    z += 1
                pred_train = agent.predict(batch)
                t_acc.append(accuracy(pred_train, labels))
            if i % 10 == 9:
                t_loss, t_acc, v_acc = t_loss / 10, np.mean(np.array(t_acc)), np.mean(np.array(v_acc))
                eval = {"val_acc": v_acc, "train_acc": t_acc, "train_loss": t_loss}
                tensorboard_eval.write_episode_data(i + epoch*280, eval)
                print(f"Epoch {i + 1 + epoch*iteration_count} --> Training Loss: {t_loss}, Training Accuracy: {t_acc} "
                      f"Validation Accuracy: {v_acc}")
                t_loss = 0
                t_acc, v_acc = [], []

    model_dir = agent.save(os.path.join(model_dir, "agent.pt"))
    print(f"Model saved in file: {model_dir}")

