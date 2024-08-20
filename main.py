import random

import torch

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
