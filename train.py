#reference : https://github.com/wjaskowski/dnnothello

from operator import pos
from cnn import *
import gzip
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
from load_data import get_train_test_val

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

if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainloader, valloader, testloader = get_train_test_val(device=device)

    #training the model
    net = CNN5Conv().to(device)
    # net = test_CNN().to(device)
    model_name = "5conv-3fc-bn"
    model_path = "/models/" + model_name + "/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    csv_path = model_path + model_name + ".csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            writer = csv.writer(f)    
            writer.writerow(["epoch", "loss", "val_accuracy"])  

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    epoch = 20

    for e in range(epoch):
        running_loss = 0.0
        csv_data = []
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, torch.max(labels, 1)[1])
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10000 == 9999:    # print every few mini batches
                correct = 0
                total = 0
                with torch.no_grad():
                    for data in valloader:
                        images, labels = data
                        outputs = net(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == torch.max(labels, 1)[1]).sum().item()
                print('[%d, %5d] loss: %.3f, val_accuracy: %.3f' %
                    (e, i + 1, running_loss / 10000, 100*correct/total))
                csv_data.append([e, running_loss/10000, 100*correct/total])
                running_loss = 0.0
        with open(csv_path, "a") as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        net.save(model_path + model_name + "-epoch-{}.pth".format(e))
    # net.save(root+"test_model.pth")

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.max(labels, 1)[1]).sum().item()
    print("Test accuracy : {}%".format(100*correct/total))

