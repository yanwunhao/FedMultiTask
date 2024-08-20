import os, json
from collections import defaultdict


def read_dir(data_dir):
    x_data = []
    y_data = []
    c_data = []

    clients = []
    data = defaultdict(None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith(".json")]

    for f in files:
        full_path = os.path.join(data_dir, f)
        with open(full_path, "r") as f_content:
            cdata = json.load(f_content)
            clients.extend(cdata["users"])
            data.update(cdata["user_data"])

    for client in clients:
        client_data = data[client]
        if len(client_data["x"]) == len(client_data["y"]):
            client_data_length = len(client_data["x"])
            for i in range(client_data_length):
                x_data.append(client_data["x"][i])
                y_data.append(client_data["y"][i])
                c_data.append(client)
        else:
            print("Fuck!")
            return

    if len(x_data) == len(y_data) == len(c_data):
        return x_data, y_data, c_data
    else:
        print("Fuck!")
        return


def read_data(train_data_dir, test_data_dir):
    train_x, train_labels, train_clients = read_dir(train_data_dir)
    test_x, test_labels, test_clients = read_dir(test_data_dir)

    return train_x, train_labels, train_clients, test_x, test_labels, test_clients


if __name__ == "__main__":
    read_dir("./data/femnist/train")
