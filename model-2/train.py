import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as utilits
from model import Nura
from config import *
from preprocess import train_dataset

model=Nura()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)
loader=utilits.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

if __name__=="__main__":
    for epoch in range(epochs):

        model.train()

        running_loss = 0.0
        num_batches = 0

        for inputs,labels in loader:

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_loss = running_loss / num_batches

        if (epoch + 1) % 50 == 0:

            print(
                f"Epoch {epoch + 1}/{epochs} "
                f"| Train MSE: {epoch_loss:.4f} "
                f"| Train RMSE: {np.sqrt(epoch_loss):.4f}"
            )


    torch.save(
        model.state_dict(),
        "model-2/model_2.pth"
    )