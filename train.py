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
    return x_train[:10000], y_train[:10000], x_valid[:500], y_valid[:500]


def preprocessing(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, save: bool = False
                  , history_length: int = 1):
    # x_train = image_processing(x_train)
    # x_valid = image_processing(x_valid)
    x_train = rgb2gray(x_train)
    x_valid = rgb2gray(x_valid)
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
    #if save:
        #torch.save(training_dataset, os.path.join("./data"))
        #torch.save(validation_dataset, os.path.join("./data"))
        #torch.save(samples_weight, os.path.join("./data"))
    return training_dataset, validation_dataset, samples_weight


def train_model(training_dataset: TensorDataset, validation_dataset: TensorDataset, samples_weight: torch.Tensor,
                batch_size: int, n_minibatches: int, lr: float, run_name: AnyStr, model_dir: AnyStr = "./models",
                tensorboard_dir="./tensorboard",
                history_length: int = 1):

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    agent = BCAgent(history_length, lr)
    tensorboard_eval = Evaluation(tensorboard_dir, run_name, ["val_acc", "train_acc", "train_loss"])

    sampler = torch.utils.data.WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight))
    train_loader = DataLoader(training_dataset, sampler=sampler, batch_size=batch_size)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    t_loss, t_acc, v_acc, v_count = 0, 0, 0, 0
    for epoch in range(3):
        for i, (batch, labels) in enumerate(train_loader):
            loss = agent.update(batch, labels)
            t_loss += loss
            with torch.no_grad():
                for val_batch, val_labels in validation_loader:
                    pred_val = agent.predict(val_batch)
                    v_acc += accuracy(pred_val, val_labels)*pred_val.shape[0]
                    v_count += pred_val.shape[0]
                pred_train = agent.predict(batch)
                t_acc += accuracy(pred_train, labels)
            if i % 10 == 9:
                t_loss, t_acc, v_acc = t_loss / 10, t_acc / 10, v_acc/v_count
                eval = {"val_acc": v_acc, "train_acc": t_acc, "train_loss": t_loss}
                tensorboard_eval.write_episode_data(i + epoch*280, eval)
                print(f"Epoch {i + 1 + epoch*280} --> Training Loss: {t_loss}, Training Accuracy: {t_acc} "
                      f"Validation Accuracy: {v_acc}")
                t_loss, t_acc, v_acc = 0, 0, 0
                v_count = 0

    model_dir = agent.save(os.path.join(model_dir, "agent_5.pt"))
    print(f"Model saved in file: {model_dir}")

