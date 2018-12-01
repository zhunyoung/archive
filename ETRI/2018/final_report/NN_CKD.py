import argparse
import numpy as np

from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import arff
from random import shuffle

class CKDDataset(Dataset):
    def __init__(self, data, label):
        self.data = np.asarray(data).astype(np.float32)
        self.label = np.asarray(label).astype(np.int64)

    def __getitem__(self, index):
        return (self.data[index], self.label[index])

    def __len__(self):
        return self.label.size

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(24, 12)
        self.fc2 = nn.Linear(12, 8)
        self.final = nn.Linear(8, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.final(x)
        return nn.Softmax(1)(x)

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(target), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.cross_entropy(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CKD DL')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=80, metavar='N',
                        help='input batch size for testing (default: 80)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 150')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print("Preprocessing...")
    data_path = "data/shuffle_63.arff"
    with open(data_path, "r") as file:
        ckd = arff.load(file.read())

    # shuffle(ckd["data"])
    # with open("data/shuffle.arff", 'w') as file:
    #     arff.dump(ckd, file)

    data = []
    for patient in ckd["data"]:
        proc_patient = []
        for i, item in enumerate(patient):
            if item is None:
                proc_patient.append(0)
            elif ckd["attributes"][i][0] == "class":
                if item == "ckd":
                    proc_patient.append(1)
                else:
                    proc_patient.append(0)
            elif isinstance(item, str):
                proc_patient.append(ckd["attributes"][i][1].index(item) + 1)
            else:
                proc_patient.append(item)
        data.append(proc_patient)
    data_np = np.array(data)
    print("Preprocessing Done.")

    # for idx, list in enumerate(data):
    #     print("patient([" + ",".join([str(x) for x in list[0:-1]]) + "]," + str(list[-1]) +").")

    train_dataset = CKDDataset(data_np[0:320, 0:-1], data_np[0:320, -1])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = CKDDataset(data_np[320:, 0:-1], data_np[320:, -1])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

    model = Net()
    optimizer = optim.Adam(model.parameters())
    acc_history = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        acc = test(args, model, device, test_loader)
        acc_history.append(acc)
    print("Average Test Accuracy : {}%".format(sum(acc_history) / len(acc_history)))