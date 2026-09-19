import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
from model import Nura
from config import *
from preprocess import test_data,test_RUL

model=Nura()
state=torch.load("model-1/model_1.pth")
model.load_state_dict(state_dict=state)
criterion=nn.MSELoss()
model.eval()

with torch.no_grad():

    test_outputs = model(test_data)

    test_loss = criterion(
        test_outputs,
        test_RUL
        )

    test_rmse = torch.sqrt(test_loss)

    print()
    print("Final Results")
    print("----------------------")
    print(f"Test MSE  : {test_loss.item():.4f}")
    print(f"Test RMSE : {test_rmse.item():.4f}")
    print(f"Test true RMSE : {41.799973*test_rmse.item():.4f}")

