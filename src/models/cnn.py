import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, nc, fs):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(nc, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(fs, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        b = x.shape[0]
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(b, -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
