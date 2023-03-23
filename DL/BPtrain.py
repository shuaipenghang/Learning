import torch
import pickle
from pathlib import Path
import gzip
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim

DATA_PATH = Path("data")
PATH = DATA_PATH/"mnist"
FILENAME = "mnist.pkl.gz"

#解压读数据
with gzip.open((PATH/FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

#转变类型
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
#可学习的参数最好用nn.Module，其他尽量用nn.functional
'''
loss_func = F.cross_entropy
weights = torch.randn([784, 10], dtype=torch.float, requires_grad=True)
bias = torch.zeros(10, requires_grad=True)
bs = 64
xb = x_train_test[0:bs]
yb = y_train_test[0:bs]
def model(xb):
    return xb.mm(weights) + bias
print(loss_func(model(xb), yb))
'''
class Mnist_NN(nn.Module):
    def __init__(self):
        super(Mnist_NN, self).__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5) #按照多少百分比

    def forward(self, x): #前向传播由自己定义
        x = F.relu(self.hidden1(x))
        x = self.dropout(x)
        x = F.relu(self.hidden2(x))
        x = self.dropout(x)
        x = self.out(x)

        return x
bs = 64
net = Mnist_NN
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs*2)

#数据封装
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs*2)
    )

def get_model():
    model = Mnist_NN()
    return model, optim.SGD(model.parameters(), lr=0.001) #SGD为梯度下降，选择更新的参数，学习率

def loss_batch(model, loss_func, xb, yb, opt = None):
    loss = loss_func(model(xb), yb) #计算损失
    if opt is not None:
        loss.backward()
        opt.step() #更新
        opt.zero_grad() #torch会进行累加，所以需要清零

    return loss.item(), len(xb)

def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval() #验证
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl] #*为解包
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('当前step:' + str(step), '验证集损失：'+str(val_loss))

loss_func = F.cross_entropy
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(20, model, loss_func, opt, train_dl, valid_dl)


correct = 0.0
total = 0.0
for xb, yb in valid_dl:
    outputs = model(xb)
    _, predicted = torch.max(outputs.data, 1)
    total += yb.size(0)
    correct += (predicted==yb).sum().item()

print("Accuracy of the network on the 10000 test image: %d %%" % (100 * correct / total))