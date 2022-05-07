import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, history_length=5, n_classes=5):
        super(CNN, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.sequential_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*9*9, 128),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.sequential(x)
        x = self.sequential_fc(x)
        return x

