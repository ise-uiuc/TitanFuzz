import torch
from torch import nn

m = nn.Dropout(p=0.2)
input = torch.randn(20, 16)
output = m(input)
