import torch
import torch.nn as nn


class attention(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.score=nn.Linear(hidden_size,1)

    def forward(self,x):
        score=self.score(x)
        score=score.squeeze(-1)
        weights=torch.softmax(score,dim=1)
        context = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1)
        return context,weights

class conv(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv=nn.Conv1d(23,64,3,1,1)
        self.batch1=nn.BatchNorm1d(64)

    def forward(self,x):
        x=x.permute(0,2,1)
        x=self.conv(x)
        x=self.batch1(x)
        x=x.permute(0,2,1)
        return x

class Nura(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv=conv()

        self.attention=attention(128)

        self.lstm=nn.LSTM(64,128,batch_first=True)

        self.fc1 = nn.Linear(128,64)

        self.fc2 = nn.Linear(64,1)



    def forward(self, x):
        x=self.conv(x)
        x=torch.relu(x)

        output, (hidden, cell) = self.lstm(x)
        context,weights=self.attention(output)        
        x = torch.relu(self.fc1(context))

        x = self.fc2(x)

        return x,weights

