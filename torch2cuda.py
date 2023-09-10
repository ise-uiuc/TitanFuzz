import argparse
import ast
import os
from types import FunctionType

import astunparse

from util import astpasses, util

try:
    import numpy as np
    import torch
except Exception as e:
    pass

try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")  # Suppress TF warnings
    # https://github.com/tensorflow/tensorflow/issues/56927
    # Run with export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda/
    import numpy as np
except Exception as e:
    pass

SEED: int = 420
OUTPUT_LIMIT: int = 1024
ALLCLOSE_RTOL: float = 1e-2  # default 1e-5
ALLCLOSE_ATOL: float = 1e-3  # default 1e-8


class PassManager:
    def __init__(self) -> None:
        pass

    def apply(self, node: ast.AST, gpu: bool = False, chkRand: bool = False) -> ast.AST:
        return None


class Config:
    def __init__(self) -> None:
        self.passManager: PassManager = self.genPassManager()

    @staticmethod
    def genPassManager() -> PassManager:
        return None

    def applyPasses(
        self, node: ast.AST, gpu: bool = False, chkRand: bool = False
    ) -> ast.AST:
        return self.passManager.apply(node, gpu, chkRand)

    @staticmethod
    def doInternalRandCheck() -> bool:
        return False

    @staticmethod
    def allclose(lhs, rhs) -> bool:
        return True

    @staticmethod
    def genExecGlobals() -> dict:
        return {}

    @staticmethod
    def isCrash(exceptMsg: str) -> bool:
        return False

    @staticmethod
    def isGpuOom(exceptMsg: str) -> bool:
        return False

    @staticmethod
    def skipApi(api: str, label: str) -> bool:
        return False


config: Config = None


class ConfigTorch(Config):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def genPassManager() -> PassManager:
        class PassManagerTorch(PassManager):
            def __init__(self) -> None:
                self.generalPasses = []
                self.cudaPasses = []
                self.chkRandPasses = []

                passRemoveImports = astpasses.PassRemoveImports()
                self.generalPasses.append(passRemoveImports)

                passRemoveNoiseCalls = astpasses.PassRemoveCalls(
                    [
                        "print",
                        "exit",
                        "torch.save",
                        "torch.manual_seed",
                        "torch.set_default_tensor_type",
                        "torch.cuda.set_device",
                        "torch.autograd.set_detect_anomaly",
                        "torch.cuda.is_available",
                        "torch.cuda.memory_allocated",
                        "torch.set_grad_enabled",
                    ]
                )
                self.generalPasses.append(passRemoveNoiseCalls)

                passFlattenCall = astpasses.PassFlattenCall()
                self.generalPasses.append(passFlattenCall)

                passRemoveTorchCuda = astpasses.PassRemoveTorchCuda()
                self.generalPasses.append(passRemoveTorchCuda)

                passReplaceEmptyTensorCalls = astpasses.PassReplaceCalls(
                    {
                        "torch.empty": "torch.randn({})",
                        "torch.set_grad_enabled": "torch.set_grad_enabled(True)",
                    }
                )
                self.generalPasses.append(passReplaceEmptyTensorCalls)

                passReplaceRawTensorsWithNoArgs = astpasses.PassReplaceCallsIfArgsEmpty(
                    {
                        "torch.Tensor": "torch.randn(3, 3)",
                        "torch.FloatTensor": "torch.randn(3, 3)",
                        "torch.DoubleTensor": "torch.randn(3, 3)",
                        "torch.HalfTensor": "torch.randn(3, 3)",
                        "torch.BFloat16Tensor": "torch.randn(3, 3)",
                        "torch.ByteTensor": "torch.randint(0, 128, (3, 3))",
                        "torch.CharTensor": "torch.randint(0, 128, (3, 3))",
                        "torch.ShortTensor": "torch.randint(0, 65536, (3, 3))",
                        "torch.IntTensor": "torch.randint(0, 1048576, (3, 3))",
                        "torch.LongTensor": "torch.randint(0, 1048576, (3, 3))",
                        "torch.BoolTensor": "torch.randint(0, 1, (3, 3))",
                    }
                )
                self.generalPasses.append(passReplaceRawTensorsWithNoArgs)

                passReplaceCallsIfArgsWithoutNameOrList = (
                    astpasses.PassReplaceCallsIfArgsWithoutNameOrList(
                        {
                            "torch.Tensor": "torch.randn({})",
                            "torch.FloatTensor": "torch.randn({})",
                            "torch.DoubleTensor": "torch.randn({})",
                            "torch.HalfTensor": "torch.randn({})",
                            "torch.BFloat16Tensor": "torch.randn({})",
                            "torch.ByteTensor": "torch.randint(0, 128, ({},))",
                            "torch.CharTensor": "torch.randint(0, 128, ({},))",
                            "torch.ShortTensor": "torch.randint(0, 65536, ({},))",
                            "torch.IntTensor": "torch.randint(0, 1048576, ({},))",
                            "torch.LongTensor": "torch.randint(0, 1048576, ({},))",
                            "torch.BoolTensor": "torch.randint(0, 1, ({},))",
                        }
                    )
                )
                self.generalPasses.append(passReplaceCallsIfArgsWithoutNameOrList)

                passReplaceMeths = astpasses.PassReplaceMeths(
                    {
                        "numpy": "cpu().numpy()",
                        "cuda": "",
                        "cpu": "",
                        "new": "clone().detach()",
                        "new_empty": "new_ones({})",
                    }
                )
                self.generalPasses.append(passReplaceMeths)

                passReplaceRandLikeCalls = astpasses.PassReplaceCalls(
                    {
                        "torch.rand_like": "torch.rand_like({}.cpu())",
                        "torch.randn_like": "torch.randn_like({}.cpu())",
                    }
                )
                self.generalPasses.append(passReplaceRandLikeCalls)

                passReplaceInplaceRandMeths = astpasses.PassReplaceMeths(
                    {
                        "random_": "",
                        "uniform_": "",
                    }
                )
                self.generalPasses.append(passReplaceInplaceRandMeths)

                passLogTorchIntermediate = astpasses.PassLogTorchIntermediate()
                self.generalPasses.append(passLogTorchIntermediate)

                passReplaceAnyCpuTensorType = astpasses.PassReplaceAny(
                    {
                        "torch.FloatTensor": "torch.cuda.FloatTensor",
                        "torch.DoubleTensor": "torch.cuda.DoubleTensor",
                        "torch.HalfTensor": "torch.cuda.HalfTensor",
                        "torch.BFloat16Tensor": "torch.cuda.BFloat16Tensor",
                        "torch.ByteTensor": "torch.cuda.ByteTensor",
                        "torch.CharTensor": "torch.cuda.CharTensor",
                        "torch.ShortTensor": "torch.cuda.ShortTensor",
                        "torch.IntTensor": "torch.cuda.IntTensor",
                        "torch.LongTensor": "torch.cuda.LongTensor",
                        "torch.BoolTensor": "torch.cuda.BoolTensor",
                    }
                )
                self.cudaPasses.append(passReplaceAnyCpuTensorType)

                passAppendTorchCuda = astpasses.PassAppendTorchCuda()
                self.cudaPasses.append(passAppendTorchCuda)

                passCheckTorchInternalRandom = astpasses.PassCheckTorchInternalRandom()
                self.chkRandPasses.append(passCheckTorchInternalRandom)

            def apply(
                self, node: ast.AST, gpu: bool = False, chkRand: bool = False
            ) -> ast.AST:
                for subpass in (
                    self.generalPasses + self.cudaPasses + self.chkRandPasses
                ):
                    subpass.reset()

                for subpass in self.generalPasses:
                    node = subpass.visit(node)
                if gpu:
                    for subpass in self.cudaPasses:
                        node = subpass.visit(node)
                if chkRand:
                    for subpass in self.chkRandPasses:
                        node = subpass.visit(node)
                return node

        return PassManagerTorch()

    @staticmethod
    def doInternalRandCheck() -> bool:
        return True

    @staticmethod
    def allclose(lhs, rhs) -> bool:
        if isinstance(lhs, torch.Tensor):
            return torch.allclose(
                lhs.cpu(),
                rhs.cpu(),
                rtol=ALLCLOSE_RTOL,
                atol=ALLCLOSE_ATOL,
                equal_nan=True,
            )
        elif isinstance(lhs, int) or isinstance(lhs, float):
            return torch.allclose(
                torch.Tensor([lhs]),
                torch.Tensor([rhs]),
                rtol=ALLCLOSE_RTOL,
                atol=ALLCLOSE_ATOL,
                equal_nan=True,
            )
        elif isinstance(lhs, torch.Size):
            return lhs == rhs
        return True

    @staticmethod
    def genExecGlobals() -> dict:
        return {"torch": torch, "np": np}

    @staticmethod
    def isCrash(exceptMsg: str) -> bool:
        return "INTERNAL ASSERT FAILED" in exceptMsg

    @staticmethod
    def isGpuOom(exceptMsg: str) -> bool:
        return False

    @staticmethod
    def skipApi(api: str, label: str) -> bool:
        return False


class ConfigTf(Config):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def genPassManager() -> PassManager:
        class PassManagerTf(PassManager):
            def __init__(self) -> None:
                self.generalPasses = []
                self.cpuPasses = []
                self.gpuPasses = []
                self.chkRandPasses = []

                passRemoveImports = astpasses.PassRemoveImports()
                self.generalPasses.append(passRemoveImports)

                passRemoveCalls = astpasses.PassRemoveCalls(
                    [
                        # remove print
                        "print",
                        "tf.print",
                        # make sure nothing disables eager mode during execution
                        # once disabled, it is hard to re-enable it in the middle of the program
                        "tf.test.main",
                        "tf.compat.v1.disable_eager_execution",
                        "tf1.disable_v2_behavior",
                        "tf.compat.v1.disable_v2_behavior",
                        "tf.compat.v1.Graph",
                        "tf.compat.v1.InteractiveSession",
                        # forbid file operations
                        "open",
                        "tf.compat.v1.keras.experimental.export_saved_model",
                        "tf.compat.v1.saved_model.experimental.save",
                        "model.save_weights",
                        "tf.io.write_file",
                        "tf.summary.create_file_writer",
                        # exit
                        "os._exit",
                    ]
                )
                self.generalPasses.append(passRemoveCalls)

                # Flattening is not a must for TF
                # passFlattenCall = astpasses.PassFlattenCall()
                # self.generalPasses.append(passFlattenCall)

                # passRandomizeTfInput = astpasses.PassRandomizeTfInput()
                # self.generalPasses.append(passRandomizeTfInput)

                passLogTfIntermediate = astpasses.PassLogTfIntermediate()
                self.generalPasses.append(passLogTfIntermediate)

                passAddTfEagerCheck = astpasses.PassAddTfEagerCheck()
                self.generalPasses.append(passAddTfEagerCheck)

                passWithTfDeviceCpu = astpasses.PassWithTfDevice("/cpu:0")
                self.cpuPasses.append(passWithTfDeviceCpu)

                passWithTfDeviceGpu = astpasses.PassWithTfDevice("/gpu:0")
                self.gpuPasses.append(passWithTfDeviceGpu)

                passCheckTorchInternalRandom = astpasses.PassCheckTorchInternalRandom()
                self.chkRandPasses.append(passCheckTorchInternalRandom)

            def apply(
                self, node: ast.AST, gpu: bool = False, chkRand: bool = False
            ) -> ast.AST:
                for subpass in (
                    self.generalPasses
                    + self.cpuPasses
                    + self.gpuPasses
                    + self.chkRandPasses
                ):
                    subpass.reset()

                for subpass in self.generalPasses:
                    node = subpass.visit(node)
                if gpu:
                    for subpass in self.gpuPasses:
                        node = subpass.visit(node)
                else:
                    for subpass in self.cpuPasses:
                        node = subpass.visit(node)
                if chkRand:
                    for subpass in self.chkRandPasses:
                        node = subpass.visit(node)
                return node

        return PassManagerTf()

    @staticmethod
    def doInternalRandCheck() -> bool:
        return False

    @staticmethod
    def allclose(lhs, rhs) -> bool:
        if isinstance(lhs, tf.Tensor):
            return np.allclose(
                lhs, rhs, rtol=ALLCLOSE_RTOL, atol=ALLCLOSE_ATOL, equal_nan=True
            )
        elif isinstance(lhs, int) or isinstance(rhs, float):
            return np.allclose(
                tf.convert_to_tensor(lhs),
                tf.convert_to_tensor(rhs),
                rtol=ALLCLOSE_RTOL,
                atol=ALLCLOSE_ATOL,
                equal_nan=True,
            )
        return True

    @staticmethod
    def genExecGlobals() -> dict:
        return {"tf": tf, "np": np, "os": os}

    @staticmethod
    def isCrash(exceptMsg: str) -> bool:
        # if ConfigTf.isGpuOom(exceptMsg): return False
        crash_kws = [
            "InternalError",
            "SystemError",
        ]
        return any([kw.lower() in exceptMsg.lower() for kw in crash_kws])

    @staticmethod
    def isGpuOom(exceptMsg: str) -> bool:
        allow_errors = [
            "Attempting to perform BLAS operation using StreamExecutor without BLAS support",
            "CUDA_ERROR_INVALID_HANDLE",
            "Could not satisfy device specification",
            "Failed to create cuFFT batched plan with scratch allocator",
        ]
        if any([allow_err.lower() in exceptMsg.lower() for allow_err in allow_errors]):
            return True
        return False

    @staticmethod
    def skipApi(api: str, label: str) -> bool:
        # These failed to be catched.
        random_apis = [
            # Random sampling
            "tf.raw_ops.Multinomial",
            "tf.keras.backend.get_uid",
        ]
        unstable_apis = [
            # matrix decomposition, multi results
            "tf.raw_ops.Svd",
            # hangs
            "tf.raw_ops.CollectiveReduce",
        ]
        examined_apis = [
            # False positives examined.
            "tf.compat.v1.gather",
            "tf.gather",
            "tf.bitwise.right_shift",
            "tf.compat.v1.bitwise.right_shift",
            "tf.compat.v1.keras.activations.relu",
            "tf.experimental.numpy.isclose",
            "tf.experimental.numpy.isreal",
            "tf.bitwise.left_shift",
            "tf.compat.v1.bitwise.left_shift",
            "tf.raw_ops.LeftShift",
            "tf.keras.layers.Wrapper",
            # OOM
            "tf.experimental.numpy.conjugate",
            "tf.signal.irfft3d",
        ]
        if api in random_apis + unstable_apis + examined_apis:
            return True
        if any([x in label for x in random_apis]):
            return True
        skip_unstable_random_kws = ["random", "svd", "segment_max", "segmentmax", "fft"]
        for kw in skip_unstable_random_kws:
            if kw in api.lower():
                return True

        # Timeout labels:
        # tf-depth
        # if label in [
        #     "tf.image.total_variation_48",
        # ]: return True
        return False


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tf", action="store_true", default=False
    )  # tensorflow or torch

    parser.add_argument("--mode", type=str, default="single")

    # for single mode
    parser.add_argument("--input", type=str, default=None)  # input .py filename
    parser.add_argument(
        "--code", action="store_true", default=False
    )  # emit transformed code
    parser.add_argument(
        "--srcast", action="store_true", default=False
    )  # emit source AST
    parser.add_argument(
        "--ast", action="store_true", default=False
    )  # emit transformed AST
    parser.add_argument(
        "--cpu", action="store_true", default=False
    )  # do not insert cuda()
    parser.add_argument(
        "--chkrand", action="store_true", default=False
    )  # check internal randomness
    parser.add_argument(
        "--noexec", action="store_true", default=False
    )  # do not execute transformed code
    parser.add_argument(
        "--noresult", action="store_true", default=False
    )  # do not print result
    parser.add_argument("--seed", type=int, default=SEED)  # random seed to use

    # for batch mode
    # --input # input source directory
    # --code # emit transformed code
    # --srcast # emit source AST
    # --ast # emit transformed AST
    # --cpu # do not insert cuda()
    # --chkrand # check internal randomness
    # --noexec # do not execute transformed code
    # --noresult # do not print result
    # --seed # random seed to use
    parser.add_argument("--start", type=int, default=0)  # from which testcase to start
    parser.add_argument(
        "--singleapi", action="store_true", default=False
    )  # stop when meeting the next api

    # for dual mode
    # --input # input .py filename

    # for race mode
    # --input # input source directory
    # --start # from which testcase to start
    # --singleapi # stop when meeting the next api

    # for duel mode
    # --input # input .py filename

    args = parser.parse_args()

    global config
    if args.tf:
        config = ConfigTf()
    else:
        config = ConfigTorch()

    if args.mode == "single":
        modeSingle(args)  # coreSingle + frameworkSingle
    elif args.mode == "batch":
        modeBatch(args)  # coreSingle + frameworkSrcBatch
    elif args.mode == "dual":
        modeDual(args)  # coreDual + frameworkSingle
    elif args.mode == "race":
        modeRace(args)  # coreDual + frameworkSrcBatch
    elif args.mode == "duel":
        modeDuel(args)  # coreDuel (contains coreDual) + frameworkSingle
    else:
        raise NotImplementedError("{} mode".format(args.mode))


def execSingle(seed: int, srcAst: ast.AST) -> dict:
    """Compiles and executes one ast, returns globals produced during its execution.
    Throws unformatted exception."""

    util.set_seed(seed)
    execGlobals = config.genExecGlobals()

    exec(compile(astunparse.unparse(srcAst), "", "exec"), execGlobals)
    execGlobals = util.removeInternalGlobals(execGlobals)
    return execGlobals


def frameworkSingle(args: argparse.Namespace, coreFunc: FunctionType) -> None:
    ifs = open(args.input, "r", encoding="utf-8")
    src: str = ifs.read()
    try:
        coreFunc(args.seed, args, src)
    except Exception as e:
        reason: str = "FrameworkCrashCatch"
        detail: str = str(e)
        if len(e.args) >= 2:
            reason: str = e.args[0]
            detail: str = e.args[1]

        print("\nFrameworkSingle", reason, SEED, detail)


def frameworkSrcBatch(args: argparse.Namespace, coreFunc: FunctionType) -> None:
    """Framework for sequential runs from source folder, compatible with driver.
    Handles arguments --start, --singleapi."""
    tasks = util.readAllTasksFromDir(args.input)
    lastApi: str = None
    for id in range(args.start, len(tasks)):
        task = tasks[id]
        api, label, src = util.parseTask(task)

        if args.singleapi:
            # One run only for the same seed
            if lastApi != None and lastApi != api:
                break
            lastApi = api

        try:
            if config.skipApi(api, label):
                raise Exception("Skipped", "no detail")
            coreFunc(SEED, args, src)

        except Exception as e:
            reason: str = "FrameworkCrashCatch"
            detail: str = str(e)
            if len(e.args) >= 2:
                reason: str = e.args[0]
                detail: str = e.args[1]
            if len(detail) > OUTPUT_LIMIT:
                detail = "Detail is too long"

            if (
                reason == "FrameworkCrashCatch"
            ):  # FrameworkCrashCatch is printed by driver
                print(detail)
                exit(-1)

            if "Catch" in reason:
                with open("catches.log", "a") as f:
                    f.write(
                        "\nTitanFuzzTestcase {} {} {} {} {} {}".format(
                            id, api, label, reason, SEED, detail
                        )
                    )
            print("\nTitanFuzzTestcase", id, api, label, reason, SEED, detail)


def coreSingle(seed: int, args: argparse.Namespace, src: str):
    srcAst = ast.parse(src)

    if args.srcast:
        print(astunparse.dump(srcAst))

    srcAst = config.applyPasses(srcAst, gpu=not args.cpu, chkRand=args.chkrand)

    if args.code:
        print(astunparse.unparse(srcAst))

    if args.ast:
        print(astunparse.dump(srcAst))

    if not args.noexec:
        try:
            execGlobals = execSingle(args.seed, srcAst)
            globalTypes = util.getTypeDict(execGlobals)
            if not args.noresult:
                util.printPretty(globalTypes)
                util.printPretty(execGlobals)
        except Exception as e:
            raise Exception("ExecFail", str(e))
        raise Exception("Success", "succeeded")


def modeSingle(args: argparse.Namespace) -> None:
    frameworkSingle(args, coreSingle)


def modeBatch(args: argparse.Namespace) -> None:
    frameworkSrcBatch(args, coreSingle)


def coreDual(seed: int, args: argparse.Namespace, src: str) -> None:
    """Throws structured exception"""

    # for seedOffset in range(1): # Try different seeds until some problem is found
    # seed = SEED + seedOffset

    # CPU

    cpuAst = ast.parse(src)
    cpuAst = config.applyPasses(cpuAst, gpu=False)
    cpuExcept = None
    try:
        cpuGlobals = execSingle(seed, cpuAst)
    except Exception as e:
        cpuExcept = e
    cpuExceptMsg: str = type(cpuExcept).__name__ + " " + str(cpuExcept)
    if config.isCrash(cpuExceptMsg):
        raise Exception("CpuCrashCatch", cpuExceptMsg)

    # GPU

    gpuAst = ast.parse(src)
    gpuAst = config.applyPasses(gpuAst, gpu=True)
    gpuExcept: Exception = None
    try:
        gpuGlobals = execSingle(seed, gpuAst)
    except Exception as e:
        gpuExcept = e
    gpuExceptMsg: str = type(gpuExcept).__name__ + " " + str(gpuExcept)
    if config.isGpuOom(gpuExceptMsg):
        raise Exception("GpuOomFail", gpuExceptMsg)
    if config.isCrash(gpuExceptMsg):
        raise Exception("GpuCrashCatch", gpuExceptMsg)

    # state compare

    if cpuExcept != None and gpuExcept != None:
        # Both failed, should be the problem with CPU AST passes
        if (
            cpuExceptMsg == gpuExceptMsg
            or cpuExceptMsg == gpuExceptMsg.replace(".cuda", "")
            or cpuExceptMsg == gpuExceptMsg.replace("cuda", "cpu")
        ):
            raise Exception(
                "BothExecFail", "\nCPU: {}\nGPU: {}".format(cpuExceptMsg, gpuExceptMsg)
            )
        elif "NotImplementedError" in gpuExceptMsg:
            raise Exception(
                "GpuNotImplementedFail",
                "\nCPU: {}\nGPU: {}".format(cpuExceptMsg, gpuExceptMsg),
            )
        elif "SyntaxError" in cpuExceptMsg and "SyntaxError" in gpuExceptMsg:
            raise Exception(
                "SyntaxFail", "\nCPU: {}\nGPU: {}".format(cpuExceptMsg, gpuExceptMsg)
            )
        else:
            raise Exception(
                "ExceptMsgCatch",
                "\nCPU: {}\nGPU: {}".format(cpuExceptMsg, gpuExceptMsg),
            )
    elif cpuExcept == None and gpuExcept != None:
        # Only GPU failed, should be the problem with GPU AST passes
        raise Exception("GpuExecFail", gpuExceptMsg)
    elif cpuExcept != None and gpuExcept == None:
        # GPU passed but CPU failed, strange enough to be considered a catch
        raise Exception("ExecStateCatch", cpuExceptMsg)

    # value compare

    cpuTypes = util.getTypeDict(cpuGlobals)
    gpuTypes = util.getTypeDict(gpuGlobals)
    cpuTypesStr: str = util.pretty(cpuTypes)
    gpuTypesStr: str = util.pretty(gpuTypes)
    if cpuTypesStr != gpuTypesStr:
        detail: str = "\nCPU:\n{}\nGPU:\n{}".format(cpuTypesStr, gpuTypesStr)
        raise Exception("VarTypeConflictCatch", detail)

    inconsistentNames = []
    for name in cpuGlobals.keys():
        cpuVal = cpuGlobals[name]
        gpuVal = gpuGlobals[name]
        try:
            if not config.allclose(cpuVal, gpuVal):
                inconsistentNames.append(name)
        except Exception as e:
            raise Exception("ComparisonFail", str(e))

    if len(inconsistentNames) == 0:
        raise Exception("Success", "succeeded")

    # Check for internal randomness before reporting catch
    hasInternalRandomness = False
    if config.doInternalRandCheck():
        try:
            chkAst = ast.parse(src)
            chkAst = config.applyPasses(chkAst, gpu=True, chkRand=True)
            execSingle(seed, chkAst)
            chkHash = torch.randn(3, 3, device="cuda:0")
            # If no random numbers are consumed by internal randomness,
            # chkHash should be as if generated before execution
            util.set_seed(seed)
            ansHash = torch.randn(3, 3, device="cuda:0")
            hasInternalRandomness = not config.allclose(chkHash, ansHash)
        except Exception as e:
            raise Exception("RandCheckExecFail", str(e))

    if hasInternalRandomness:
        raise Exception("InternalRandomFail", "")

    cpuGlobalsNumeric = util.removeNonNumericGlobals(cpuGlobals)
    gpuGlobalsNumeric = util.removeNonNumericGlobals(gpuGlobals)
    detail: str = ""
    try:
        detail: str = "\ndiff:{}\nCPU:\n{}\nGPU:\n{}".format(
            util.pretty(inconsistentNames),
            util.pretty(cpuGlobalsNumeric),
            util.pretty(gpuGlobalsNumeric),
        )
    except Exception as e:
        raise Exception("VarInconsistentCatch", "Unable to print values " + str(e))
    raise Exception("VarInconsistentCatch", detail)


def modeDual(args: argparse.Namespace) -> None:
    frameworkSingle(args, coreDual)


def modeRace(args: argparse.Namespace) -> None:
    frameworkSrcBatch(args, coreDual)


def coreDuel(seed: int, args: argparse.Namespace, src: str):
    srcLines: list[str] = src.split("\n")
    lo: int = 0
    hi: int = len(srcLines)
    lastReason = "no reason"
    lastDetail = "no detail"
    while hi > lo:
        mid: int = (lo + hi) // 2
        partialSrc: str = ""
        for srcLine in srcLines[0 : mid + 1]:
            partialSrc += srcLine + "\n"

        try:
            coreDual(seed, args, partialSrc)

        except Exception as e:
            reason: str = "FrameworkCrashCatch"
            detail: str = str(e)
            if len(e.args) >= 2:
                reason: str = e.args[0]
                detail: str = e.args[1]
            if reason == "Success":
                lo = mid + 1
            else:
                hi = mid
                lastReason = reason
                lastDetail = detail
    if lo == len(srcLines):
        raise Exception("DuelFailed", "No problem found")

    # Print last lines before problem
    problemLines: str = "Last lines before problem: \n"
    for i in range(max(hi - 3, 0), hi + 1):
        problemLines += "{} > {}\n".format(i + 1, srcLines[i])

    raise Exception(
        "DuelFinished",
        "Problem since line {}\n{}{} {}".format(
            hi + 1, problemLines, lastReason, lastDetail
        ),
    )


def modeDuel(args: argparse.Namespace) -> None:
    frameworkSingle(args, coreDuel)


if __name__ == "__main__":
    main()
    # Some sneaky code may contain exit(0) or other equivalent calls
    # We distinguish ourselves from them with a magic number
    exit(233)
