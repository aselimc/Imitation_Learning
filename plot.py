import matplotlib.pyplot as plt
import os
import gzip
import numpy as np
import pickle
from utils import action_to_id
from tensorflow.python.summary.summary_iterator import summary_iterator



def data_loader(dataset_dir="./data"):
    data_file = os.path.join(dataset_dir, "data.pkl.gzip")
    data = pickle.load(gzip.open(data_file, "rb"))
    x = np.array(data["state"]).astype("float32")
    y = np.array(data["action"]).astype("float32")
    y = np.array([action_to_id(a) for a in y])
    return x, y


def label_counter(y):
    counts = np.unique(y, return_counts=True)
    return counts


def hist(x, y, x_label, y_label, title):
    fig, axs = plt.subplots(2, figsize=(6,6))
    axs[0].bar(x, y)
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel(y_label)
    weights = 1. / y
    axs[1].bar(x, weights)
    axs[1].set_xlabel("Actions")
    axs[1].set_ylabel("Probabilities of instances")
    fig.savefig("./figures/weighted_random_sampler.svg")

def extract(path):
    loss = []
    t_acc = []
    v_acc = []
    for e in summary_iterator(path):
        for v in e.summary.value:
            if v.tag == "train_loss":
                loss.append(v.simple_value)
            elif v.tag == "train_acc":
                t_acc.append(v.simple_value)
            elif v.tag == "val_acc":
                v_acc.append(v.simple_value)
    return np.array(loss).astype("float32"), np.array(t_acc).astype("float32"), np.array(v_acc).astype("float32")

def plotter(loss, t_acc, v_acc):
    iteration = np.arange(t_acc.shape[0])
    fig, axs = plt.subplots(2, figsize=(8, 8))
    axs[0].plot(iteration*10, loss, label="Training Loss")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")
    axs[0].grid()
    axs[1].plot(iteration*10, t_acc, label="Training Accuracy")
    axs[1].plot(iteration*10, v_acc, label="Validation Accuracy")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Accuracy")
    plt.legend()
    plt.grid()
    fig.savefig("./figures/ImitationLResults.svg")

if __name__ == "__main__":
    x, y = data_loader("./data")
    labels, counts = label_counter(y)
    hist(labels, counts, "Actions", "Counts", "Initial Distribution")
    extracted = extract(r"C:\Users\Ahmet\DEEP_LEARNING\Imitation_Learning\tensorboard\hist_5-20220510-024734\events.out.tfevents.1652143654.tfpool24.2338380.0")
    plotter(*extracted)