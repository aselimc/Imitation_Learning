import numpy as np
from typing import List
import torch

STRAIGHT = 0
LEFT = 1
RIGHT = 2
ACCELERATE = 3
BRAKE = 4
global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    This method converts rgb images to grayscale.
    Args:
        rgb: .shape = (batch_size, height, weight, 3)

    Returns:
    Grayscale output with shape (batch_size, height, weight)
    """
    gray = np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721]).astype("float32")
    return gray


def image_processing(x: np.ndarray):
    if len(x.shape) == 3:
        x = single_image_processing(x)
        x = rgb2gray(x)
        return x
    else:
        for index, image in enumerate(x):
            x[index] = single_image_processing(image)
        x = rgb2gray(x)
        return x


def coloring(image, old, new):
    mask = np.all(image == old, axis=2)
    image[mask] = new
    return image


def single_image_processing(image: np.ndarray) -> np.ndarray:
    image[85:, :18] = [0, 0, 0]
    image = coloring(image, [000., 000., 000, ], [50., 50., 50.])
    # Coloring Road n Curves
    image = coloring(image, [102., 102., 102.], [150., 150., 150.])
    image = coloring(image, [103., 103., 103.], [150., 150., 150.])
    image = coloring(image, [104., 104., 104.], [150., 150., 150.])
    image = coloring(image, [105., 105., 105.], [150., 150., 150.])
    image = coloring(image, [106., 106., 106.], [150., 150., 150.])
    image = coloring(image, [107., 107., 107.], [150., 150., 150.])
    image = coloring(image, [255., 255., 255.], [255., 000., 000.])
    # Coloring Grass
    image = coloring(image, [102., 229., 102.], [150., 250., 100.])
    image = coloring(image, [102., 204., 102.], [150., 250., 100.])
    image = coloring(image, [102., 217., 102.], [150., 250., 100.])
    image = coloring(image, [198., 168., 073.], [150., 250., 100.])
    image = coloring(image, [198., 231., 198.], [150., 250., 100.])
    image = coloring(image, [255., 188., 188.], [255., 000., 000.])
    return image


def action_to_id(a: List) -> int:
    """
    This method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    Returns: Integers
    """
    if all(a == [-1.0, 0.0, 0.0]):
        return LEFT  # LEFT: 1
    elif all(a == [1.0, 0.0, 0.0]):
        return RIGHT  # RIGHT: 2
    elif all(a == [0.0, 1.0, 0.0]):
        return ACCELERATE  # ACCELERATE: 3
    elif np.allclose(a, [0.0, 0.0, 0.2], 1e-8):
        return BRAKE  # BRAKE: 4
    else:
        return STRAIGHT  # STRAIGHT = 0


def id_to_action(action_id: int, max_speed: int = 0.8) -> np.ndarray:
    """
    This method makes actions continuous.
    Important: this method only works if you recorded data pressing only one key at a time!
    Args:
        action_id: Action classification as integer
        max_speed: Top speed

    Returns:
    List of cont. actions
    """
    if action_id == LEFT:
        return np.array([-1.0, 0.0, 0.01])
    elif action_id == RIGHT:
        return np.array([1.0, 0.0, 0.01])
    elif action_id == ACCELERATE:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, 0.1])
    else:
        return np.array([0.0, 0.0, 0.0])


def history_stack(x: torch.Tensor, history_length: int) -> torch.Tensor:
    bs = x.shape[0]
    init = 0
    latter = init + history_length
    extended_array = np.array([x[0]] * (history_length - 1))
    extended_array = np.concatenate((extended_array, x)) if history_length > 1 else x
    del x
    new_x = np.empty(shape=(bs, history_length, 96, 96))
    while latter != len(extended_array):
        new_x[init] = np.stack(extended_array[init:latter])
        init += 1
        latter += 1
    del extended_array
    new_x = torch.HalfTensor(new_x).to(device)
    return new_x


def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = predictions.detach().cpu().argmax(1)
    labels = labels.detach().cpu()
    scores = torch.zeros_like(labels)
    scores[labels == predictions] = 1
    return torch.sum(scores) / len(scores)
