import torch
from torch import nn
import d2l

net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 5*5卷积层， 生成六个
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),  # 2*2汇聚层
    nn.Conv2d(6, 16, kernel_size=5),  # 输出六个
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),  # 对张量展平
    nn.Linear(16 * 5 * 5, 120),  # 全连接层， 此处输入被压缩为5 * 5
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

batch_size = 64
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size = batch_size)
