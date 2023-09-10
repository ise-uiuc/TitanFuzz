import torch

try:
    x = torch.tensor([-1.0]).cuda()
    y = torch.tensor([2.0]).cuda()
    loss = torch.nn.BCELoss().cuda()
    output = loss(x, y)  # Causes CUDA internal assertion failure
except Exception as e:
    print("init: exception caught: {}".format(e))
else:
    print("init: no exception")

for i in range(3):
    try:
        a = torch.tensor([-1.0]).cuda()
        b = torch.tensor([2.0]).cuda()
        c = a + b  # Raises CUDA exception again
    except Exception as e:
        print("{}: exception caught: {}".format(i, e))
    else:
        print("{}: no exception".format(i))
