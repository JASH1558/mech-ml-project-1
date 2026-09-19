import torch
import torch.nn as nn

class conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv1d(23,64,3,1,1)
        self.batch=nn.BatchNorm1d(64)
    def forward(self,x):
        x=x.permute(0,2,1)
        x=self.conv(x)
        x=self.batch(x)
        x=x.permute(0,2,1)
        return x

class Nura(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv=conv()
        self.lstm=nn.LSTM(64,128,batch_first=True)

        self.fc1 = nn.Linear(128,64)

        self.fc2 = nn.Linear(64,1)


    def forward(self, x):
        x=self.conv(x)
        x=torch.relu(x)

        output, (hidden, cell) = self.lstm(x)
        last_output=output[:,-1,:]
        
        x=self.fc1(last_output)
        x = torch.relu(x)
        x = self.fc2(x)

        return x


