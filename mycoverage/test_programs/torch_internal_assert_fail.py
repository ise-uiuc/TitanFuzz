import torch


def fn(input):
    return input.aminmax(dim=0)


device = "cuda"

input = torch.empty([36, 0], dtype=torch.float32)
input.uniform_(-16, 3)

fn(input.to(device))
# RuntimeError: iter.numel() > 0 && iter.ntensors() - iter.noutputs() == 1 && iter.noutputs() >= 1 INTERNAL ASSERT FAILED at "/opt/conda/conda-bld/pytorch_1663571524876/work/aten/src/ATen/native/cuda/Reduce.cuh":1134, please report a bug to PyTorch.
