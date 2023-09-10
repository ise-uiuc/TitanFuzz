import argparse
import json
import os
import subprocess
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np

from mycoverage import mp_executor
from util.util import (
    ExecutionStatus,
    run_cmd,
    wrap_code_with_device,
    wrap_code_with_mock,
)

CURRENT_TIME = time.time()


def validate_status(
    code, library, validate_mode="process", device="cpu", api=None, test_executor=None
) -> (ExecutionStatus, str):
    code = wrap_code_with_mock(code, api, library, device)
    if validate_mode == "process":
        return validate_status_process(code, library, device=device)
    elif validate_mode == "exec":
        return validate_status_exec(code, library, device=device)
    elif validate_mode == "multiprocess":
        return validate_status_mp(
            code, library, mp_executor.test_executor, device=device
        )


def validate_status_exec(g_code, library, device="gpu") -> (ExecutionStatus, str):
    """Executing code via exec.

    Note that this is dangerous.
    Directly executing unknown code could overwrite some important files.
    """
    write_code = wrap_code_with_device(g_code, library, device)
    with open("/tmp/tmp{}.py".format(CURRENT_TIME), "w") as f:
        f.write(write_code)
    try:
        if library == "tf":
            import tensorflow as tf

            execGlobals = {"tf": tf, "np": np}
        else:
            import torch

            execGlobals = {"torch": torch, "np": np}
        exec(write_code, execGlobals)
    except Exception as e:
        error_msg = type(e).__name__ + " " + str(e)
        if "TargetNotCalledError" in error_msg:
            return ExecutionStatus.NOTCALL, error_msg
        return ExecutionStatus.EXCEPTION, error_msg
    else:
        return ExecutionStatus.SUCCESS, ""


def validate_status_mp(
    g_code, library, test_executor, device="cpu"
) -> (ExecutionStatus, str):
    CURRENT_TIME = time.time()
    tmp_filename = "/tmp/tmp{}.py".format(CURRENT_TIME)
    write_code = wrap_code_with_device(g_code, library, device)
    with open(tmp_filename, "w") as f:
        f.write(write_code)
    # with open("temp.py", "w") as f:
    #     f.write(write_code)
    status_msg, valid = test_executor.run_test(tmp_filename)
    status_msg_, valid_ = test_executor.check_if_internal_state_break()

    if valid and valid_:
        return ExecutionStatus.SUCCESS, ""
    else:
        if "timeout" in status_msg:
            return ExecutionStatus.TIMEOUT, ""
        elif "INTERNAL ASSERT FAILED".lower() in status_msg.lower():
            return ExecutionStatus.CRASH, status_msg
        elif "TargetNotCalledError" in status_msg:
            return ExecutionStatus.NOTCALL, status_msg
        elif "exception" in status_msg:
            return ExecutionStatus.EXCEPTION, status_msg
        elif "crash" in status_msg:
            return ExecutionStatus.CRASH, status_msg
        elif "Error" in status_msg:
            return ExecutionStatus.CRASH, status_msg
        else:
            return ExecutionStatus.EXCEPTION, status_msg


def validate_status_process(
    g_code, library, python="python", device="cpu", verbose=False
) -> (ExecutionStatus, str):
    write_code = wrap_code_with_device(g_code, library, device)
    with open("/tmp/tmp{}.py".format(CURRENT_TIME), "w") as f:
        f.write(write_code)
    run_args = [python, "/tmp/tmp{}.py".format(CURRENT_TIME)]
    status, msg = run_cmd(run_args, verbose=verbose)
    if "TargetNotCalledError" in msg:
        return ExecutionStatus.NOTCALL, msg
    else:
        return status, msg


def validate_status_docker(
    g_code, library, device="cpu", mount=True
) -> (ExecutionStatus, str):
    write_code = wrap_code_with_device(g_code, library, device)
    with open("/tmp/tmp{}.py".format(CURRENT_TIME), "w") as f:
        f.write(write_code)

    run_args = "docker run --gpus all -it --rm -v /tmp:/tmp tensorflow/tensorflow:latest-gpu python /tmp/tmp{}.py".format(
        CURRENT_TIME
    ).split(
        " "
    )

    print(run_args)

    return run_cmd(run_args, verbose=False)


def validate(g_code, library):
    with open("/tmp/tmp{}.py".format(CURRENT_TIME), "w") as f:
        if library == "torch":
            f.write("import torch\n")
        elif library == "tf":
            f.write("import tensorflow as tf\n")
        f.write("import numpy as np\n")
        f.write(g_code)
    try:
        output = subprocess.run(
            "python /tmp/tmp{}.py".format(CURRENT_TIME),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            shell=True,
        )
        print("output.returncode: ", output.returncode)
        if output.returncode != 0:
            return False
        else:
            return True
    except subprocess.TimeoutExpired as te:
        print("Timed out")
        return False


def validate_outputs(args):
    with open(args.file, "r") as f:
        outputs = json.load(f)

    validation_time = 0
    num_valid = 0
    val_ret = dict()
    print(" Total APIs: ", len(outputs))
    device = "gpu"
    for api_name, record in outputs.items():
        seed_code = record["seed"]
        mutants = record["outputs"]

        start = time.time()
        initial_status, error_msg = validate_status(
            seed_code,
            args.library,
            validate_mode=args.validate_mode,
            device=args.device,
        )
        initial = initial_status == ExecutionStatus.SUCCESS
        end = time.time()

        validation_time += end - start
        val_ret[api_name] = dict()
        val_ret[api_name]["seed"] = seed_code
        val_ret[api_name]["initial_status"] = str(initial_status)
        val_ret[api_name]["error_msg"] = error_msg
        val_ret[api_name]["initial"] = initial
        mut_ret = []

        for mutant in mutants:
            code = mutant["output"]
            diff = mutant["diff"]
            num = mutant["num"]
            start = time.time()
            status, error_msg = validate_status(code, args.library)
            valid = status == ExecutionStatus.SUCCESS
            end = time.time()

            validation_time += end - start
            num_valid += valid
            mut_ret.append(
                {
                    "output": code,
                    "diff": diff,
                    "num": num,
                    "status": str(status),
                    "error_msg": error_msg,
                    "valid": valid,
                }
            )
        val_ret["api_name"]["outputs"] = mut_ret

    print("{} valid outputs using {}s validation".format(num_valid, validation_time))
    return val_ret, validation_time


def recursive_clean(original: str, library: str) -> str:
    """If syntax error / run fail, then remove last line until success."""
    code_lines = original.split("\n")
    code = "\n".join(code_lines)
    while True:
        status, errmsg = validate_status_exec(code, library)
        if status == ExecutionStatus.SUCCESS:
            return code
        code_lines = code_lines[:-1]
        code = "\n".join(code_lines)
        if code == "":
            break
    return code


def main(args):

    if args.library not in ["torch", "tf"]:
        raise NotImplementedError

    os.makedirs(args.output_dir, exist_ok=True)

    if args.library == "tf":
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        from util.util import set_memory_growth

        set_memory_growth()
    elif args.mode == "validate":
        val_ret, val_time = validate_outputs(args)
        with open(args.output, "w") as f:
            json.dump(val_ret, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--library", type=str, default=None, help="either 'torch' or 'tf'"
    )
    parser.add_argument("--api", type=str, default=None)
    parser.add_argument("--id", type=int, default=None)
    parser.add_argument("--code", action="store_true", default=False)  # print code
    parser.add_argument("--verbose", action="store_true", default=False)  # print code
    parser.add_argument(
        "--validate_mode",
        type=str,
        default="process",
        choices=["process", "exec", "multiprocess"],
    )
    parser.add_argument("--device", type=str, default="gpu")

    parser.add_argument(
        "--docker",
        action="store_true",
        default=False,
        help="if set true, start a docker container to run the experiment.",
    )
    parser.add_argument("--close_fd_mask", type=int, default=1)
    args = parser.parse_args()

    print("---- BEGIN VALIDATE ----")
    print("args:\n{}".format(args))

    if args.validate_mode == "multiprocess":
        mp_executor.init_test_executor(args)
        print("Finish initializing...")
    main(args)

    mp_executor.kill_executors()
