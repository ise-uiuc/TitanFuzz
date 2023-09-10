import multiprocessing
import os

import torch

x = torch.Tensor(
    [
        2,
    ]
).cuda()
y = torch.Tensor(
    [
        2,
    ]
).cuda()
torch.add(x, y)

from mycoverage.mp_executor import (
    Executor,
    PyTorchCoverageExecutor,
    PyTorchExecutor,
    TensorFlowCoverageExecutor,
    TensorFlowExecutor,
)

_test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_programs")


def test_tf_cov():
    executor = TensorFlowCoverageExecutor(single_test_timeout=10, close_fd_mask=3)

    # Test: corrupt state (disable eager), exception, crash are handled gracefully
    files = [
        "tf1.py",
        "corrupt_state.py",
        "tf1.py",
        "crash.py",
        "test.py",
        "tf1.py",
        "tf2.py",
        "exception.py",
        "tf2.py",
    ]
    # Test: Release memory after the next program call
    # files = ["largemodel.py",
    # "loopforever.py",
    # "loopforever.py",
    # "loopforever.py",
    # "loopforever.py",
    # "loopforever.py",
    # "loopforever.py",
    # ]

    # Test: basic
    # files = ["tf1.py", "tf2.py"]

    for f in files:
        ret = executor.run_test(os.path.join(_test_dir, f))
        if f in ["tf1.py", "tf2.py"]:
            assert ret[0] == "ok"
        ret = executor.check_if_internal_state_break()
    executor.terminate()
    print("test_tf_cov passed")


def test_tf():
    executor = TensorFlowExecutor(single_test_timeout=10, close_fd_mask=3)

    # Test: corrupt state (disable eager), exception, crash are handled gracefully
    files = [
        "tf1.py",
        "corrupt_state.py",
        "tf1.py",
        "crash.py",
        "test.py",
        "tf1.py",
        "tf2.py",
        "exception.py",
        "tf2.py",
    ]

    for f in files:
        ret = executor.run_test(os.path.join(_test_dir, f))
        if f in ["tf1.py", "tf2.py"]:
            assert ret[0] == "ok"
        ret = executor.check_if_internal_state_break()
    executor.terminate()
    print("test_tf passed")


def test_torch():
    executor = PyTorchExecutor(single_test_timeout=10, close_fd_mask=3)

    files = [
        "torch1.py",
        "torch_largemodel.py",
        "torch_timeout.py",
        "torch_timeout.py",
        "torch_timeout.py",
        "torch2.py",
        "torch_cuda_fail.py",
        "torch1.py",
    ]

    for f in files:
        print(f)
        ret = executor.run_test(os.path.join(_test_dir, f))
        if f in ["torch1.py", "torch2.py"]:
            assert ret[0] == "ok"
            assert ret[1] == True
        elif f == "torch_cuda_fail.py":
            # raise exception, does not crash
            assert ret[1] == True

        ret = executor.check_if_internal_state_break()
        if f in ["torch1.py", "torch2.py"]:
            assert ret[0] == "ok"
            assert ret[1] == True
        elif f == "torch_cuda_fail.py":
            assert "CUDA error" in ret[0]
            assert not ret[1]

    executor.terminate()
    print("test_torch passed")


def test_torch_cov():
    executor = PyTorchCoverageExecutor(single_test_timeout=10, close_fd_mask=3)

    files = ["torch1.py", "torch2.py", "torch_cuda_fail.py", "torch1.py"]

    for f in files:
        ret = executor.run_test(os.path.join(_test_dir, f))
        if f in ["torch1.py", "torch2.py"]:
            assert ret[0] == "ok"
        elif f == "torch_cuda_fail.py":
            # raise exception, does not crash
            assert ret[1] == True

        ret = executor.check_if_internal_state_break()
        if f in ["torch1.py", "torch2.py"]:
            assert ret[0] == "ok"
        elif f == "torch_cuda_fail.py":
            assert "CUDA error" in ret[0]
            assert not ret[1]

    executor.terminate()
    print("test_torch_cov passed")


if __name__ == "__main__":

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    # test_tf()
    test_torch()

    # Can only trace one library (tf or torch) at a time
    # test_tf_cov()
    # test_torch_cov()
