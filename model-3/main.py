
import torch
import torch.nn as nn
import numpy as np
data=np.loadtxt("CMaps/test_FD001.txt")

data=data[data[:,0]==8]
final=data[-1][1]
data=data[30:60]
data = data[:, 1:]
data=torch.tensor(data,dtype=torch.float32)
data=data.reshape(-1,1)
batch_size=32
input_variables=25
window_size=30
input_layer_neurons=25*window_size
hidden_layer_neurons_one=64
hidden_layer_neurons_two=128
hidden_layer_neurons_three=32
output_layer_neurons=1  
class nura(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(input_layer_neurons,hidden_layer_neurons_one)
        self.fc2=nn.Linear(hidden_layer_neurons_one,hidden_layer_neurons_two)
        self.fc3=nn.Linear(hidden_layer_neurons_two,hidden_layer_neurons_three)
        self.fc4=nn.Linear(hidden_layer_neurons_three,output_layer_neurons)
    def forward(self,x):
        x=x.reshape(x.shape[0],-1)
        x=self.fc1(x)
        x=torch.relu(x)
        x=self.fc2(x)
        x=torch.relu(x)
        x=self.fc3(x)
        x=torch.relu(x)
        x=self.fc4(x)
        return x


model=nura()
dicttt=torch.load("model")
model.load_state_dict(dicttt)
model.eval()
with torch.no_grad():
    output=model(data.T)

print(output)
print(final)