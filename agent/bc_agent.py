import torch
from network import CNN

global device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BCAgent:

    def __init__(self, history_length: int, lr: float):
        self.net = CNN(history_length).to(device)
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr)

    def update(self, x: torch.Tensor, y: torch.Tensor):
        self.optimizer.zero_grad()
        predictions = self.predict(x)
        loss = self.criterion(predictions, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x: torch.Tensor):
        if len(x.shape) == 3:  # Single Image
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        if not torch.is_tensor(x):
            x = torch.Tensor(x).to(device)
        return self.net(x)

    def load(self, f_name):
        self.net.load_state_dict(torch.load(f_name))

    def save(self, f_name):
        torch.save(self.net.state_dict(), f_name)
