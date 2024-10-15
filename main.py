import random

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from util.dataset import FEMNIST
from model.Nets import CNNFemnist

RANDOM_SEED = 12345

if __name__ == "__main__":

    random.seed(RANDOM_SEED)

    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )

    print("Running on", device)

    training_data = FEMNIST(train=True)
    test_data = FEMNIST(train=False)

    train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False)

    model = CNNFemnist("")
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # training
    num_epochs = 50
    for round in range(num_epochs):
        max_value = 0
        for batch_idx, (x, y, c) in enumerate(test_dataloader):

            pred = model(x)
            print(pred.shape)

        break
