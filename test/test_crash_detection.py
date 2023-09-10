import argparse
import os
import time

from termcolor import colored

from mycoverage import mp_executor
from mycoverage.mp_executor import (
    PyTorchCoverageExecutor,
    PyTorchExecutor,
    TensorFlowCoverageExecutor,
    TensorFlowExecutor,
    coverate_run_status_mp,
)
from validate import ExecutionStatus, validate_status

test_executor = None
cov_executor = None


def test_tf_executor(args):
    global test_executor
    test_executor = TensorFlowExecutor(
        single_test_timeout=10, close_fd_mask=args.close_fd_mask, cpu=True
    )

    testfns = [
        "tf1.py",
        "crash.py",
        "tf2.py",
        "exception.py",
        "tf1.py",
        "loopforever.py",
        "tf2.py",
    ]
    test_dir = "mycoverage/test_programs"
    for fn in testfns:
        print(fn)
        with open(os.path.join(test_dir, fn), "r") as fr:
            code = fr.read()
        start = time.time()
        status, msg = validate_status(
            code, "tf", validate_mode="multiprocess", test_executor=test_executor
        )
        valid = status == ExecutionStatus.SUCCESS
        end = time.time()

        print(status, msg)
        print("time: ", end - start)

    test_executor.kill()


def test_tf_cov_fuzz(args):
    global test_executor
    global cov_executor
    test_executor = TensorFlowExecutor(
        single_test_timeout=10, close_fd_mask=args.close_fd_mask, cpu=True
    )
    cov_executor = TensorFlowCoverageExecutor(
        single_test_timeout=10, close_fd_mask=args.close_fd_mask, cpu=True, debug=True
    )

    testfns = [
        "tf1.py",
        "crash.py",
        "tf2.py",
        "exception.py",
        "tf1.py",
        "loopforever.py",
        "tf2.py",
        "tf1.py",
        "tf2.py",
        "tf1.py",
    ]
    test_dir = "mycoverage/test_programs"
    cnt_tf1 = 0
    for fn in testfns:
        print(fn)
        with open(os.path.join(test_dir, fn), "r") as fr:
            code = fr.read()
        status, msg = validate_status(
            code, "tf", validate_mode="multiprocess", test_executor=test_executor
        )
        valid = status == ExecutionStatus.SUCCESS
        print(colored("trial run {} {} {}".format(fn, str(status), msg), "green"))
        if fn in ["tf1.py", "tf2.py"]:
            assert valid
        if valid:
            status_, new_coverage = coverate_run_status_mp(
                code, "tf", cov_executor=cov_executor
            )
            assert status_ == ExecutionStatus.SUCCESS
            print(
                colored(
                    " >> coverage run {} {} {}".format(fn, str(status_), new_coverage),
                    "green",
                )
            )
            if fn == "tf1.py":
                cnt_tf1 += 1
                if cnt_tf1 > 2:
                    # when run for the first time: 512 -> 1928
                    # when run for the second time, the coverage can still increase: 1947 -> 1960
                    # however, after the second time, coverage should not increase.
                    assert not new_coverage
        end = time.time()

        print(status, msg)

    test_executor.kill()
    cov_executor.kill()


def test_torch_cov_fuzz(args):
    mp_executor.init_test_executor(args, True)

    files = [
        "torch1.py",
        "torch_internal_assert_fail.py",
        "torch_crash.py",
        "torch2.py",
        "torch_cuda_fail.py",
        "torch1.py",
        "torch_timeout.py",
        "torch1.py",
        "torch2.py",
    ]
    test_dir = "mycoverage/test_programs"
    for fn in files:
        with open(os.path.join(test_dir, fn), "r") as fr:
            code = fr.read()
        start = time.time()
        status, msg = validate_status(
            code,
            args.library,
            validate_mode="multiprocess",
            test_executor=mp_executor.test_executor,
        )
        valid = status == ExecutionStatus.SUCCESS
        print(colored("trial run {} {} {}".format(fn, str(status), msg), "green"))

        if fn in ["torch1.py", "torch2.py"]:
            assert valid
        elif fn == "torch_cuda_fail.py":
            # raise exception, does not crash
            assert not valid
        elif fn == "torch_internal_assert_fail.py":
            assert status == ExecutionStatus.CRASH
        if valid:
            status_, new_coverage = coverate_run_status_mp(
                code, args.library, cov_executor=mp_executor.cov_executor
            )
            assert status_ == ExecutionStatus.SUCCESS
            print(
                colored(
                    " >> coverage run {} {} {}".format(fn, str(status_), new_coverage),
                    "green",
                )
            )

    mp_executor.kill_executors()
    print("test_torch_cov passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment setup configs
    parser.add_argument(
        "--library", type=str, default="tf", help="either 'torch' or 'tf'"
    )
    parser.add_argument("--close_fd_mask", type=int, default=1)
    parser.add_argument("--cov", action="store_true", default=False)
    args = parser.parse_args()
    try:
        if args.library == "tf":
            if not args.cov:
                test_tf_executor(args)
            else:
                test_tf_cov_fuzz(args)
        elif args.library == "torch":
            if args.cov:
                test_torch_cov_fuzz(args)
    except Exception as e:
        print("exception in main: ", e, type(e).__name__)
        if test_executor is not None:
            test_executor.kill()
