import json
import logging
import multiprocessing
import multiprocessing as mp
import os
import sys
import time

from mycoverage import tracer
from util.util import ExecutionStatus, wrap_code_with_device

# By default execute in cpu mode
RUN_CPU_MODE = True
test_executor = None
cov_executor = None


def init_test_executor(args, cov=False):
    global test_executor
    global cov_executor
    kwargs = {}
    if hasattr(args, "close_fd_mask"):
        kwargs["close_fd_mask"] = args.close_fd_mask
    if hasattr(args, "debug"):
        kwargs["debug"] = args.debug

    if args.library == "torch":
        if cov and (cov_executor is None):
            # kwargs['debug'] = True
            cov_executor = PyTorchCoverageExecutor(**kwargs)
        if test_executor is None:
            test_executor = PyTorchExecutor(**kwargs)

    if args.library == "tf":
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        from util.util import set_memory_growth

        set_memory_growth()

        kwargs["cpu"] = RUN_CPU_MODE
        if cov and (cov_executor is None):
            # kwargs['debug'] = True
            cov_executor = TensorFlowCoverageExecutor(**kwargs)
        if test_executor is None:
            test_executor = TensorFlowExecutor(**kwargs)


def kill_executors():
    if test_executor is not None:
        test_executor.kill()
    if cov_executor is not None:
        cov_executor.kill()


def exec_func(filename, execGlobals=None):
    # Code should contains imports
    import numpy as np
    import tensorflow as tf

    with open(filename, "r") as f:
        code = f.read()
    if execGlobals is None:
        exec(code)
    else:
        exec(code, execGlobals)


def exec_func_tf_cpu(filename, execGlobals=None):
    # Code should contains imports
    import numpy as np
    import tensorflow as tf

    with open(filename, "r") as f:
        code = f.read()
    if execGlobals is None:
        with tf.device("cpu"):
            exec(code)
    else:
        with tf.device("cpu"):
            exec(code, execGlobals)


def cov_worker_tf(target, child_conn, close_fd_mask):

    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.CRITICAL)
    f = open(os.devnull, "w")
    if close_fd_mask & 1:
        sys.stdout = f
    if close_fd_mask & 2:
        sys.stderr = f

    sys.settrace(tracer.trace_tf)
    while True:
        try:
            buf = child_conn.recv_bytes().decode("utf-8")
        except EOFError:
            # child does not respond, probably crash
            message = "Crash: EOFError"
            try:
                child_conn.send_bytes(message.encode("utf-8"))
            except BrokenPipeError:
                break

        try:
            msg = target(buf)
            # exec("import tensorflow as tf\ntf.keras.backend.clear_session()")
            # print("msg", msg)
        except Exception as e:
            print("Exception: %r\n" % (e,))
            logging.exception(e)
            message = {"exception": type(e).__name__, "msg": str(e)}
            message = json.dumps(message)
            child_conn.send_bytes(message.encode("utf-8"))
        else:
            child_conn.send_bytes(b"%d" % tracer.get_coverage())


def cov_worker_torch(target, child_conn, close_fd_mask):

    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.CRITICAL)
    f = open(os.devnull, "w")
    if close_fd_mask & 1:
        sys.stdout = f
    if close_fd_mask & 2:
        sys.stderr = f

    sys.settrace(tracer.trace_torch)
    while True:
        buf = child_conn.recv_bytes().decode("utf-8")
        try:
            msg = target(buf)
        except Exception as e:
            print("Exception: %r\n" % (e,))
            logging.exception(e)
            message = {"exception": type(e).__name__, "msg": str(e)}
            message = json.dumps(message)
            child_conn.send_bytes(message.encode("utf-8"))
        else:
            child_conn.send_bytes(b"%d" % tracer.get_coverage())


def worker_tf(target, child_conn, close_fd_mask):

    import numpy as np
    import tensorflow as tf

    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.CRITICAL)
    f = open(os.devnull, "w")
    if close_fd_mask & 1:
        sys.stdout = f
    if close_fd_mask & 2:
        sys.stderr = f

    while True:
        try:
            buf = child_conn.recv_bytes()
        except EOFError:
            # child does not respond, probably crash
            message = "Crash: EOFError"
            try:
                child_conn.send_bytes(message.encode("utf-8"))
            except BrokenPipeError:
                break
        try:
            execGlobals = {"tf": tf, "np": np}
            msg = target(buf, execGlobals)
        except Exception as e:
            print("Exception: %r\n" % (e,))
            logging.exception(e)
            message = {"exception": type(e).__name__, "msg": str(e)}
            message = json.dumps(message)
            child_conn.send_bytes(message.encode("utf-8"))
        else:
            child_conn.send_bytes("ok".encode("utf-8"))


def worker_torch(target, child_conn, close_fd_mask):

    import numpy as np
    import torch

    logging.captureWarnings(True)
    logging.getLogger().setLevel(logging.CRITICAL)
    f = open(os.devnull, "w")
    if close_fd_mask & 1:
        sys.stdout = f
    if close_fd_mask & 2:
        sys.stderr = f

    while True:
        try:
            buf = child_conn.recv_bytes().decode("utf-8")
        except EOFError:
            # child does not respond, probably crash
            message = "Crash: EOFError"
            try:
                child_conn.send_bytes(message.encode("utf-8"))
            except BrokenPipeError:
                break
        try:
            # execGlobals = {'torch': torch, 'numpy': np, 'np': np}
            # msg = target(buf, execGlobals)
            # torch.cuda.empty_cache()
            msg = target(buf)
        except Exception as e:
            print("Exception: %r\n" % (e,))
            logging.exception(e)
            message = {"exception": type(e).__name__, "msg": str(e)}
            message = json.dumps(message)
            child_conn.send_bytes(message.encode("utf-8"))
        else:
            child_conn.send_bytes("ok".encode("utf-8"))


class Executor:
    def __init__(
        self,
        worker,
        single_test_timeout=10,
        close_fd_mask=0,
        exec_func=exec_func,
        debug=False,
    ):
        self.worker = worker
        self._close_fd_mask = close_fd_mask
        ctx = multiprocessing.get_context("spawn")
        self.parent_conn, self.child_conn = ctx.Pipe()
        self._timeout = single_test_timeout
        self._exec_func = exec_func
        self.debug = debug
        self._p = ctx.Process(
            target=self.worker,
            args=(self._exec_func, self.child_conn, self._close_fd_mask),
        )
        self._p.start()
        self._test_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "test_programs"
        )

    def run_test(self, filename) -> (str, bool):
        """
        Returns execution status message and if it is valid
        """
        start = time.time()

        self.parent_conn.send_bytes(filename.encode("utf-8"))

        wait_secs = 0
        start_wait_time = time.time()
        try:
            while (
                self.parent_conn.poll() is False
                and self._p.is_alive()
                and wait_secs < self._timeout
            ):
                if self.parent_conn.poll(1):
                    break
                wait_secs = time.time() - start_wait_time
        except Exception as e:
            print("Exception encountered: ", e)

        if not self._p.is_alive():
            # Crash:
            return "crash", ""
        if wait_secs >= self._timeout or self.parent_conn.poll() is False:
            print("=================================================================")
            print("timeout reached. testcase took: {}".format(self._timeout))

            self.restart()
            return "timeout", False
        try:
            return_bytes = self.parent_conn.recv_bytes()
            message = return_bytes.decode("utf-8")
            valid = message == "ok"
            return message, valid
        except Exception as e:
            # print("Error value: ", return_bytes)
            return (
                "Error: {} - {} - {}".format(type(e).__name__, str(e), return_bytes),
                False,
            )

    def terminate(self):
        self._p.terminate()

    def kill(self):
        self._p.kill()

    def restart(self):
        self._p.kill()
        ctx = multiprocessing.get_context("spawn")
        self.parent_conn, self.child_conn = ctx.Pipe()
        self._p = ctx.Process(
            target=self.worker,
            args=(self._exec_func, self.child_conn, self._close_fd_mask),
        )
        self._p.start()

    def check_if_internal_state_break(self):
        # Test run a simple cuda A+B program
        # If failed, could be CUDA error: device-side assert triggered
        # In this case, restart executor
        status, valid = self.run_test(self.check_filename)
        if status != "ok":
            self.restart()
        return status, valid


class CoverageExecutor(Executor):
    def __init__(self, worker, single_test_timeout=10, **kwargs):
        super().__init__(worker, single_test_timeout, **kwargs)
        self.prev_coverage = 0
        print("Init cov executor")

    def run_test(self, filename) -> (str, bool):
        """
        Returns a string (execution status) and a boolean (if coverage increases).

        execution status: one of "timeout", "ok", "crash", "Error: .."
        """
        start = time.time()

        self.parent_conn.send_bytes(filename.encode("utf-8"))

        if not self.parent_conn.poll(self._timeout):
            print("=================================================================")
            print("timeout reached. testcase took: {}".format(self._timeout))

            s_time = time.time()
            # Unexpected error
            print(
                "[Error] ... \n"
                "Hangs during coverage collection. \n"
                "Had to restart coverage executor..."
            )
            self.restart()
            return "timeout", False
        try:
            return_bytes = self.parent_conn.recv_bytes()
            total_coverage = int(return_bytes)
            new_coverage = False
            print("Cov: {} -> {}".format(self.prev_coverage, total_coverage))
            if total_coverage > self.prev_coverage:
                if self.debug:
                    print(
                        "Cov increases: {} -> {}".format(
                            self.prev_coverage, total_coverage
                        )
                    )
                new_coverage = True
            self.prev_coverage = total_coverage
            return "ok", new_coverage
        except ValueError:
            msg = return_bytes.decode("utf-8")
            return msg, False

    def terminate(self):
        self._p.terminate()

    def check_if_internal_state_break(self):
        status, valid = self.run_test(self.check_filename)

        if status != "ok":
            # Unexpected error
            print(
                "[Error] ... \n"
                "Internal state broke during coverage collection. \n"
                "Had to restart coverage executor..."
            )
            self.restart()
        return status, valid


class TensorFlowCoverageExecutor(CoverageExecutor):
    def __init__(
        self, worker=cov_worker_tf, single_test_timeout=10, cpu=False, **kwargs
    ):
        if cpu:
            my_exec_func = exec_func_tf_cpu
        else:
            my_exec_func = exec_func
        super().__init__(worker, single_test_timeout, exec_func=my_exec_func, **kwargs)
        assert tracer.trace_library is None
        tracer.trace_library = "tf"
        self.run_test(os.path.join(self._test_dir, "set_memory_growth.py"))
        self.check_filename = os.path.join(self._test_dir, "check_tf_state.py")


class TensorFlowExecutor(Executor):
    def __init__(self, worker=worker_tf, single_test_timeout=10, cpu=False, **kwargs):
        if cpu:
            exec_func = exec_func_tf_cpu
        super().__init__(worker, single_test_timeout, exec_func=exec_func, **kwargs)
        if self.debug:
            print(" --- init tf executor: set_memory_growth --- ")
        self.run_test(os.path.join(self._test_dir, "set_memory_growth.py"))
        self.check_filename = os.path.join(self._test_dir, "check_tf_state.py")


class PyTorchExecutor(Executor):
    def __init__(self, single_test_timeout=10, **kwargs):
        super().__init__(
            worker=worker_torch, single_test_timeout=single_test_timeout, **kwargs
        )
        self.check_filename = os.path.join(self._test_dir, "check_torch_state.py")


class PyTorchCoverageExecutor(CoverageExecutor):
    def __init__(self, worker=cov_worker_torch, single_test_timeout=10, **kwargs):
        super().__init__(worker, single_test_timeout, **kwargs)
        assert tracer.trace_library is None
        tracer.trace_library = "torch"
        print("Initizalize torch cov")
        self.check_filename = os.path.join(self._test_dir, "check_torch_state.py")


def coverate_run_status_mp(
    g_code, library, cov_executor, device="cpu"
) -> (ExecutionStatus, bool):
    """
    Returns status and whether has new coverage
    """
    CURRENT_TIME = time.time()
    tmp_filename = "/tmp/tmp{}.py".format(CURRENT_TIME)
    write_code = wrap_code_with_device(g_code, library, device)
    with open(tmp_filename, "w") as f:
        f.write(write_code)

    status, new_coverage = cov_executor.run_test(tmp_filename)

    if status == "ok":
        return ExecutionStatus.SUCCESS, new_coverage
    else:
        if "timeout" in status:
            return ExecutionStatus.TIMEOUT, new_coverage
        elif "exception" in status:
            return ExecutionStatus.EXCEPTION, new_coverage
        elif "crash" in status:
            return ExecutionStatus.CRASH, new_coverage
        elif "Error" in status:
            return ExecutionStatus.CRASH, new_coverage
        else:
            return ExecutionStatus.EXCEPTION, new_coverage
