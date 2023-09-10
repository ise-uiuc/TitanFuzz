import torch

shape = [1000, 1000, 1000]
x = torch.randn(*shape).cuda()
# y = torch.randn(*shape).cuda()
# z = torch.matmul(x, y).size()
print(x.size())
while True:
    pass
