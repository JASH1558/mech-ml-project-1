import torch
import torch.nn as nn

class Nura(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc1 = nn.Linear( 23*30,64 )

        self.fc2 = nn.Linear(64,128)

        self.fc3 = nn.Linear(128,32)

        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):

        x = x.reshape(x.shape[0], -1)

        x = torch.relu(self.fc1(x))

        x = torch.relu(self.fc2(x))

        x = torch.relu(self.fc3(x))

        x = self.fc4(x)

        return x


