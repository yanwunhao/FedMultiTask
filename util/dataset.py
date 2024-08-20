import numpy as np
import torch
from torch.utils.data import Dataset

from .io import read_data


class FEMNIST(Dataset):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """

    def __init__(
        self,
        train,
        transform=None,
        target_transform=None,
    ):
        super(FEMNIST, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        train_x, train_labels, train_clients, test_x, test_labels, test_clients = (
            read_data("./data/femnist/train", "./data/femnist/test")
        )

        if self.train:
            train_data_x = []
            train_data_y = []
            train_data_c = []

            for i in range(len(train_x)):
                train_data_x.append(np.array(train_x[i]).reshape(28, 28))
                train_data_y.append(train_labels[i])
                train_data_c.append(train_clients[i])

            self.data = train_data_x
            self.label = train_data_y
            self.client = train_data_c

        else:
            test_data_x = []
            test_data_y = []
            test_data_c = []

            for i in range(len(test_x)):
                test_data_x.append(np.array(test_x[i]).reshape(28, 28))
                test_data_y.append(test_labels[i])
                test_data_c.append(test_clients[i])

            self.data = test_data_x
            self.label = test_data_y
            self.client = test_data_c

    def __getitem__(self, index):
        img, target, client = self.data[index], self.label[index], self.client[index]
        img = np.array([img])
        # img = Image.fromarray(img, mode='L')
        # if self.transform is not None:
        #     img = self.transform(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)
        return (torch.from_numpy((0.5 - img) / 0.5).float(), target, client)

    def __len__(self):
        return len(self.data)
