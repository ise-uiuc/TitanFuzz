import glob
import json
import os
import pprint
import random
import re
import subprocess
from difflib import unified_diff
from enum import IntEnum, auto
from typing import Tuple

import numpy as np

hasTorch = True
try:
    import torch
except Exception as e:
    hasTorch = False

hasTf = True
try:
    import tensorflow as tf
except Exception as e:
    hasTf = False


class ExecutionStatus(IntEnum):
    SUCCESS = auto()
    EXCEPTION = auto()
    CRASH = auto()
    NOTCALL = auto()
    TIMEOUT = auto()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if hasTorch:
        torch.cuda.empty_cache()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasTf:
        tf.random.set_seed(seed)


def get_unified_diff(source, changed_source):
    output = ""
    for line in unified_diff(
        source.split("\n"), changed_source.split("\n"), lineterm=""
    ):
        output += line + "\n"
    return output


def readTasks(path: str):
    data = {}
    with open(path) as jsonFile:
        data = json.load(jsonFile)
    tasks = []
    # scan tasks
    for api, content in data.items():
        src: str = content["seed"]
        tasks.append((api, "{}_seed".format(api), src))
        variantId = 0
        outputs = content["outputs"]
        if isinstance(outputs, list):  # old format
            for variant in outputs:
                if not variant["valid"]:
                    continue
                src: str = variant["output"]
                variantId += 1
                tasks.append((api, "{}_{}".format(api, variantId), src))
        elif isinstance(outputs, dict):  # new format from round mutation
            for src, detail in outputs.items():
                if detail["parent"] == None:
                    continue  # is seed
                variantId += 1
                tasks.append((api, "{}_{}".format(api, variantId), src))
    return tasks


def parseTask(task: list) -> Tuple[str, str, str]:
    api: str = task[0]
    label: str = task[1]
    src: str = task[2]
    return api, label, src


def readAllTasksFromDir(directory, collect_mode="all", target_api=None):
    """Returns a list of [api, label, code]"""

    tasks = []
    if os.path.exists(os.path.join(directory, "valid")):
        # Generation structure:
        # diretory |
        #     seed |
        #     valid |
        #         api1_1.py
        #         ...
        #     exception |
        #     ... |
        if collect_mode == "seed":
            subdirs = [
                "seed",
            ]
        elif collect_mode == "valid":
            subdirs = ["seed", "valid"]
        elif collect_mode == "exception":
            subdirs = ["exception"]
        else:
            subdirs = ["seed", "valid", "exception"]
        print("collect_mode:", collect_mode, "subdirs: ", subdirs)
        for subdir in subdirs:
            target_pyfile_format = (
                "*.py" if target_api is None else "{}*.py".format(target_api)
            )
            for program in glob.glob(
                os.path.join(directory, subdir, target_pyfile_format)
            ):
                with open(program, "r") as f:
                    original_code = f.read()
                label = program.split("/")[-1].rsplit(".", 1)[0]
                api = label.rsplit("_", 1)[0]
                tasks.append([api, label, original_code])

    else:
        programs = glob.glob(os.path.join(directory, "*.py"))
        if len(programs) == 0:
            # Flattened structure
            # directory |
            #     api
            #         1.py
            #         ...
            for api in glob.glob(os.path.join(directory, "*")):
                for program in glob.glob(os.path.join(api, "*.py")):
                    with open(program, "r") as f:
                        original_code = f.read()
                    api = api.split("/")[-1]
                    label = api + "_" + program.split("/")[-1].split(".")[0]
                    tasks.append([api, label, original_code])

        else:
            # Flattened structure
            # directory |
            #     api1_1.py
            #         ...
            #     api2_1.py
            for program in programs:
                label = program.split("/")[-1].rsplit(".", 1)[0]
                api = label.split("_")[0]
                with open(program, "r") as f:
                    original_code = f.read()
                tasks.append([api, label, original_code])
    tasks.sort()
    return tasks


def readTasksFromDir(directory, skip_labels=[], skip_apis=[]):
    data = {}
    tasks = []
    srcs = dict()
    for pyfile in glob.glob(os.path.join(directory, "*.py")):
        api = os.path.basename(pyfile).rsplit("_", 1)[0]
        if api not in srcs:
            srcs[api] = []
        if if_skip_api(api, "tf"):
            continue
        with open(pyfile, "r") as f:
            src = f.read()
        if src in srcs[api]:
            continue
        srcs[api].append(src)
        label = os.path.basename(pyfile).rstrip(".py")
        if label in skip_labels:
            continue
        if api in skip_apis:
            continue
        tasks.append((api, label, src))
    return tasks


def readTasksOrder(orderFile, directory, skip_labels=[]):
    with open(orderFile, "r") as f:
        labels = f.read().splitlines()
    tasks = []
    for label in labels:
        if label in skip_labels:
            continue
        api = label.rsplit("_", 1)[0]
        with open(os.path.join(directory, label + ".py"), "r") as f:
            src = f.read()
        tasks.append((api, label, src))
    return tasks


# various simple heuristics to clean code
def clean_code(output: str) -> str:
    clean_output = output
    # for prefix generation remove import torch
    if clean_output.splitlines()[0] == "import torch":
        clean_output = "\n".join(clean_output.splitlines()[1:])

    # for suffix generation
    pattern = re.compile(r"<.+>")  # such as <cell>, <test>, </cell>
    clean_output = re.split(pattern, clean_output)[0]

    return clean_output


def pretty(obj) -> str:
    return pprint.pformat(obj, indent=4)


def printPretty(obj):
    print(pretty(obj))


def getTypeDict(globalDict: dict) -> dict:
    typeDict = {}
    for name, val in globalDict.items():
        typeDict[name] = type(val).__name__
    return typeDict


with open("data/tf_skip_prefix_all_deprecated.json", "r") as f:
    tf_skip_prefix = json.load(f)


def if_skip_api(api, library):
    if library == "tf":
        return any([api.startswith(prefix) for prefix in tf_skip_prefix])
    else:
        return False


def load_apis(library, sample=False, apilist_fn=None):
    if apilist_fn is not None:
        with open(apilist_fn, "r") as f:
            apis = f.read().splitlines()
        return apis
    if sample:
        if library == "tf":
            with open("data/tf_apis_100sample.txt", "r") as f:
                apis = f.read().splitlines()
            return apis
        elif library == "torch":
            with open("data/torch_apis_100sample.txt", "r") as f:
                apis = f.read().splitlines()
            return apis

    if library == "tf":
        with open("data/tf_apis.txt", "r") as f:
            apis = f.read().splitlines()
        return apis
    elif library == "torch":
        with open("data/torch_apis.txt", "r") as f:
            apis = f.read().splitlines()
        return apis


def load_api_symbols(library):
    if library == "tf":
        with open("data/tf_apis.txt", "r") as f:
            apis = f.read().splitlines()
        api_call_list = [api.rsplit(".", 1)[-1] for api in apis]
        return api_call_list, apis
    elif library == "torch":
        with open("data/torch_apis.txt", "r") as f:
            apis = f.read().splitlines()
        api_call_list = [api.rsplit(".", 1)[-1] for api in apis]
        return api_call_list, apis


def removeInternalGlobals(globalDict: dict) -> dict:
    retDict = globalDict.copy()
    for name in globalDict:
        if name.startswith("__"):
            retDict.pop(name)
    return retDict


def removeNonNumericGlobals(globalDict: dict) -> dict:
    retDict = globalDict.copy()
    for name, val in globalDict.items():
        isNumeric = False
        if isinstance(val, list):
            isNumeric = True
        if isinstance(val, np.ndarray):
            isNumeric = True
        if hasTorch:
            if isinstance(val, torch.Tensor) or isinstance(val, torch.Size):
                isNumeric = True
        if hasTf:
            if isinstance(val, tf.Tensor):
                isNumeric = True
        if isinstance(val, int) or isinstance(val, float):
            isNumeric = True
        if not isNumeric:
            retDict.pop(name)
    return retDict


def run_cmd(
    cmd_args,
    timeout=10,
    verbose=False,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    shell=False,
) -> (ExecutionStatus, str):
    try:
        output = subprocess.run(
            cmd_args, stdout=stdout, stderr=stderr, timeout=timeout, shell=shell
        )

    except subprocess.TimeoutExpired as te:
        if verbose:
            print("Timed out")
        return ExecutionStatus.TIMEOUT, ""
    else:
        if verbose:
            print("output.returncode: ", output.returncode)
        if output.returncode != 0:
            # 134 = Crash
            # 1 = exception
            error_msg = ""
            if output.stdout is not None:
                stdout_msg = output.stdout.decode("utf-8")
                stderr_msg = output.stderr.decode("utf-8")
                if verbose:
                    print("stdout> ", stdout_msg)
                if verbose:
                    print("stderr> ", stderr_msg)
                stdout_msg = stdout_msg[:30]
                error_msg = "---- returncode={} ----\nstdout> {}\nstderr> {}\n".format(
                    output.returncode, stdout_msg, stderr_msg
                )

            if output.returncode == 134:  # Failed assertion
                return ExecutionStatus.CRASH, "SIGABRT Triggered\n" + error_msg
            elif output.returncode == 132:
                return ExecutionStatus.CRASH, "SIGILL\n" + error_msg
            elif output.returncode == 133:
                return ExecutionStatus.CRASH, "SIGTRAP\n" + error_msg
            elif output.returncode == 136:
                return ExecutionStatus.CRASH, "SIGFPE\n" + error_msg
            elif output.returncode == 137:
                return ExecutionStatus.CRASH, "OOM\n" + error_msg
            elif output.returncode == 138:
                return ExecutionStatus.CRASH, "SIGBUS Triggered\n" + error_msg
            elif output.returncode == 139:
                return (
                    ExecutionStatus.CRASH,
                    "Segmentation Fault Triggered\n" + error_msg,
                )
            else:
                if output.returncode != 1:
                    # Check Failed: -6
                    print("output.returncode: ", output.returncode)
                    print(cmd_args)
                    print("stdout> ", stdout_msg)
                    print("stderr> ", stderr_msg)
                    return ExecutionStatus.CRASH, error_msg
                else:
                    return ExecutionStatus.EXCEPTION, error_msg
        else:
            if verbose:
                stdout_msg = output.stdout.decode("utf-8")
                print("stdout> ", stdout_msg)
            return ExecutionStatus.SUCCESS, ""


def wrap_code_with_device(g_code, library, device):
    write_code = ""
    if library == "torch":
        write_code += "import torch\n"
    elif library == "tf":
        write_code += "import os\nos.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'\n"
        if device == "cpu":
            write_code += "os.environ['CUDA_VISIBLE_DEVICES']=''\n"
        write_code += "import tensorflow as tf\n"
        write_code += "tf.get_logger().setLevel('ERROR')\n"
    write_code += "import numpy as np\n"
    write_code += g_code
    return write_code


def _add_indent(code):
    return "\n".join(["    " + codeline for codeline in code.splitlines()])


def wrap_code_with_mock(g_code, api, library, device):
    if api is None:
        return g_code
    write_code = ""
    if api.startswith("torch.Tensor."):  # Mock class method differently
        method_name = api.rsplit(".", 1)[1]
        write_code += "from unittest import mock\n"
        write_code += (
            "with mock.patch.object("
            "torch.Tensor, '{}', autospec=True) as __mock_func:\n".format(method_name)
        )
        write_code += _add_indent(
            "__mock_func.return_value = '{}'\n".format(method_name)
        )
        write_code += _add_indent(g_code)
        write_code += _add_indent(
            '\nif __mock_func.call_count == 0: raise Exception("TargetNotCalledError")\n'
        )
        return write_code

    # Mock
    write_code += "__target_func = {}\n".format(api)
    write_code += "from unittest import mock\n"
    write_code += "__mock_func = mock.Mock(side_effect=__target_func)\n"
    module_name, api_name = api.rsplit(".", 1)
    write_code += 'setattr({}, "{}", __mock_func)\n'.format(module_name, api_name)

    # Run
    write_code += g_code + "\n"

    # Recover
    write_code += 'setattr({}, "{}", __target_func)\n'.format(module_name, api_name)

    # Raise TargetNotCalledError if target function is not called
    write_code += (
        'if __mock_func.call_count == 0: raise Exception("TargetNotCalledError")\n'
    )

    return write_code


def set_memory_growth():
    import tensorflow as tf

    physical_devices = tf.config.list_physical_devices("GPU")
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass


def parseResultSummary(line: str):
    try:
        if not line.startswith("TitanFuzzTestcase"):
            return None, None, None, None
        breakDown = line.split(" ")
        if len(breakDown) < 5:
            return None, None, None, None
        id: int = int(breakDown[1])
        api: str = breakDown[2]
        label: str = breakDown[3]
        state: str = breakDown[4]
        return id, api, label, state
    except Exception:
        return None, None, None, None
