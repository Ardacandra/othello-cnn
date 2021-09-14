import itertools
from operator import pos
from cnn import *
import gzip
import pickle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader

def encode_channels(board, player):
    """
    Encodes the board into 2x8x8 binary matrix.
    The first matrix has ones indicating the fields occupied by the player who is about to play.
    The second matrix has ones where the opponent's pieces are.

    Returns 2x8x8 array.
    """
    opponent = 1 if player==2 else 2
    c1 = board == player
    c2 = board == opponent
    return np.concatenate((c1[None, ...], c2[None, ...]), axis=0)


def load_data(data_path, encoder=encode_channels, shape=(2, 8, 8), zip=True):
    if zip:
        data = pickle.load(gzip.open(data_path, "rb"))
    else:
        data = pickle.load(open(data_path, "rb"))

    x = np.zeros(((len(data),) + shape), dtype=np.uint8)
    y = np.zeros(len(data), dtype=np.uint8)

    for i, (board, player, move) in enumerate(data):
        x[i] = encoder(board, player)
        y[i] = move
    return x, y

def one_hot(move, possible_moves):
    return (possible_moves==move)*1

def get_train_test_val(device, data_path="dataset/data.dump"):
    # data_path = "dataset/data_sample.pkl"
    # X, y = load_data(data_path, zip=False)
    X, y = load_data(data_path)

    positions = list(itertools.product(range(1,9), repeat=2))
    skipped = [(4, 4), (4, 5), (5, 4), (5, 5)]
    positions = [p for p in positions if p not in skipped]
    possible_moves = [a*10+b for (a,b) in positions]

    y = [one_hot(val, possible_moves) for val in y]

    #split to train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=1
    )

    #split to train, test, and validation data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=1
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.10, random_state=1
    )

    #change to tensors
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    X_val = torch.Tensor(X_val).to(device)
    y_val = torch.Tensor(y_val).to(device)
    X_test = torch.Tensor(X_test).to(device)
    y_test = torch.Tensor(y_test).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 64
    trainloader = DataLoader(train_dataset, batch_size=batch_size)
    valloader = DataLoader(val_dataset, batch_size=batch_size)
    testloader = DataLoader(test_dataset, batch_size=batch_size)

    return trainloader, valloader, testloader

def get_testloader(device, data_path="dataset/data.dump"):
    X, y = load_data(data_path)

    positions = list(itertools.product(range(1,9), repeat=2))
    skipped = [(4, 4), (4, 5), (5, 4), (5, 5)]
    positions = [p for p in positions if p not in skipped]
    possible_moves = [a*10+b for (a,b) in positions]

    y = [one_hot(val, possible_moves) for val in y]

    #split to train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=1
    )

    #change to tensors
    X_test = torch.Tensor(X_test).to(device)
    y_test = torch.Tensor(y_test).to(device)

    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 64
    testloader = DataLoader(test_dataset, batch_size=batch_size)

    return testloader