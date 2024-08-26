import random

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from util.dataset import FEMNIST

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

    train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

    # training
