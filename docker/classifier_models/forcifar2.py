import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.player1 = nn.Sequential(  # [3, 32, 32]
            nn.Conv2d(3, 6, 5),  # 卷积层 [6, 28, 28]
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(2)  # 池化  [6, 14, 14]
        )

        self.player2 = nn.Sequential(  # [6, 14, 14]
            nn.Conv2d(6, 16, 5),  # [16, 10, 10]
            nn.ReLU(),
            nn.MaxPool2d(2)  # [16, 5, 5]
        )

        self.player3 = nn.Sequential(  # 全连接层
            nn.Linear(16 * 5 * 5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):  # 传播过程
        x = self.player1(x)
        x = self.player2(x)
        x = x.reshape(x.size()[0], -1)  # 平面展开
        output = self.player3(x)

        return output

def cifarnet2(num_classes=43):
    return Net()
