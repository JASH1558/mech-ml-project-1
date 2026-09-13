import torch
import torch.nn as nn

class Nura(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm=nn.LSTM(25,128,batch_first=True)

        self.fc1 = nn.Linear(128,64)

        self.fc2 = nn.Linear(64,1)


    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        last_output=output[:,-1,:]
        x = torch.relu(self.fc1(last_output))

        x = self.fc2(x)

        return x


