深度学习步骤：
import torch.nn.functional as F
from torch import nn
from torch import optim
进入网络的数据需要进入同一个设备，如cpu或gpu
tensorflow需要使用tensor类型的数据

类的初始化中:
需要super对本身进行继承
神经网络层设定：
self.hidden1 = nn.Linear(h1, h2)
Dropout设定（此处是设置百分之五十的节点去掉）：
self.dropout = nn.Dropout(0.5)
前向传播由自己设定：
forward：
x = F.relu(self.hidden1(x))
x = self.dropout(x)

损失函数：
F.cross_entropy，输入两个参数返回损失函数
优化方法：
optim.Adam(model.parameters(), lr=)

更新步骤:
model.train()训练一次
while:
loss.backward() #损失进行反向传播
opt.setp() #更新
opt.zero_grad() #更新梯度