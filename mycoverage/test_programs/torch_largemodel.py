import torch
from torch import nn


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.m = nn.Conv2d(
            16, 333, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1)
        )

    def forward(self, x):
        return self.m(x)


model = MyModel().cuda()
input = torch.randn(20, 16, 50, 100).cuda()
output = model(input)
print(output.size())
while True:
    pass
