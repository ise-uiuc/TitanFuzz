import argparse
import glob
import json
import multiprocessing
import os
import subprocess
import time

from model import SpanLM
from mycoverage import mp_executor
from process_file import clean_code, get_initial_programs
from util.clean_code import dead_code_elim
from util.instrumentor import SnippetInfill
from util.Logger import Logger
from util.Seed_pool import GA, GAR, GA_Coverage, GA_Random, GAR_depth
from util.util import ExecutionStatus, load_apis, run_cmd, set_seed
from validate import validate_status

os.environ[
    "TOKENIZERS_PARALLELISM"
] = "false"  # disables annoying warning caused by validation (spawns new process)
CURRENT_TIME = time.time()


def generate_loop(
    args, model: SpanLM, original_codes: list, api: str, logger: Logger, max_valid: int
):
    num_selection = 1
    num_valid, num_sampled, generation_time, validation_time, total_run_time = (
        0,
        0,
        [],
        [],
        [],
    )
    (
        num_timeout,
        num_exception,
        num_crash,
        num_duplicated,
        num_notarget,
        num_generated,
    ) = (0, 0, 0, 0, 0, 0)
    total_outputs = set(original_codes)
    GA_class = GAR_depth
    if args.seed_selection_algo == "random":
        GA_class = GA_Random
    elif args.seed_selection_algo == "coverage":
        GA_class = GA_Coverage

    ga = GA_class(
        original_codes,
        num_selection,
        args.batch_size,
        args.folder,
        api,
        model.infill_ph,
        args.library,
        args.relaxargmut,
        args.seed_selection_algo,
        args.mutator_selection_algo,
        args.use_single_mutator,
        args.replace_type,
        args.seed_pool_size,
        args.mutator_set,
    )
    r = 0
    import torch

    crashes = []
    total_programs = []
    while (max_valid < 0 or num_valid < max_valid) and sum(
        total_run_time
    ) < args.timeout:
        logger.logo("--- Round : {} ---".format(r))
        start_time_total = time.time()
        round_valid = 0
        selections = ga.selection()
        g_time, v_time = 0, 0
        for seed, infill_code, replace_type in selections:
            generations = []
            filenames = []
            add_flags = []
            start = time.time()
            well, early_stop, outputs = model.model_predict_multi(
                infill_code, do_sample=True, num_samples=args.batch_size
            )
            end = time.time()
            g_time += end - start

            for output in outputs:
                output = clean_code(
                    output, prints_and_imports=True, comment=True, cuda=True
                )
                output = dead_code_elim(output, api)
                num_generated += 1
                if output in total_outputs:
                    num_duplicated += 1
                    continue
                total_outputs.add(output)

                num_replaced, _, _ = SnippetInfill(
                    mask_identifier=model.infill_ph,
                    api_call=api.split(".")[-1],
                    prefix=".".join(api.split(".")[1:-1]),
                    library=args.library,
                    replace_type="argument",
                ).add_infill(output)
                start = time.time()
                status, msg = validate_status(
                    output,
                    args.library,
                    validate_mode=args.validate_mode,
                    test_executor=mp_executor.test_executor,
                )
                valid = status == ExecutionStatus.SUCCESS
                end = time.time()
                v_time += end - start

                if num_replaced < 1:
                    # The target API could be replaced by another API
                    # for now let's also dump the code in a separate folder
                    # but we don't put it in the seed pool
                    subfolder = "notarget"
                    dump_code = '"""\n{}\n{}\n"""\n{}'.format(str(status), msg, output)
                    with open(
                        os.path.join(
                            args.folder,
                            subfolder,
                            api + "_" + str(num_generated) + ".py",
                        ),
                        "w",
                    ) as f:
                        f.write(dump_code)
                    num_notarget += 1
                    continue

                dump_code = output
                subfolder = ""
                if status == ExecutionStatus.SUCCESS:
                    subfolder = "valid"
                if status == ExecutionStatus.TIMEOUT:
                    num_timeout += 1
                    subfolder = "hangs"
                elif status == ExecutionStatus.CRASH:
                    status_, msg_ = validate_status(
                        output, args.library, validate_mode="process"
                    )
                    if status_ == ExecutionStatus.CRASH:
                        # Crash find!
                        num_crash += 1
                        subfolder = "crash"
                        crashes.append(output)
                        logger.logo("--- crash found : {}---".format(msg_))
                        dump_code = '"""\n' + msg_ + '\n"""\n' + output
                    else:
                        # the previous crash could be due to some polluted state
                        subfolder = "flaky"
                elif status == ExecutionStatus.EXCEPTION:
                    num_exception += 1
                    dump_code = '"""\n' + msg + '\n"""\n' + output
                    subfolder = "exception"

                # Dump all generated programs, including invalid ones
                filename = os.path.join(
                    args.folder, subfolder, api + "_" + str(num_generated) + ".py"
                )
                with open(filename, "w") as f:
                    f.write(dump_code)
                torch.cuda.empty_cache()

                if valid:  # not just valid but has the same format
                    round_valid += 1
                    generations.append(output)
                    filenames.append(filename)
                    if args.seed_selection_algo == "coverage":
                        status_, new_coverage = mp_executor.coverate_run_status_mp(
                            output, args.library, cov_executor=mp_executor.cov_executor
                        )
                        print("> coverage run: ", status_, new_coverage)
                        add_flags.append(new_coverage)
            if args.seed_selection_algo == "coverage":
                ga.update(seed, generations, replace_type, r, filenames, add_flags)
            else:
                ga.update(seed, generations, replace_type, r, filenames)

        num_valid += round_valid
        if round_valid == 0:  # restarts if none of the generations are valid (rare)
            mp_executor.test_executor.restart()
        generation_time.append(g_time)
        validation_time.append(v_time)
        total_programs.append(num_generated)
        r += 1
        logger.logo(
            "--- New Valid : {} using {}s generation, {}s validation ---".format(
                round_valid, g_time, v_time
            )
        )
        # cleanup
        torch.cuda.empty_cache()

        total_run_time.append(time.time() - start_time_total)

    n, highest_order = ga.get_highest_order_output()
    logger.logo("Highest Order: {}".format(highest_order))
    logger.logo("----- \n {} \n ----- ".format(n))
    logger.logo(
        "{} valid outputs using {}s generation, {}s validation".format(
            num_valid, sum(generation_time), sum(validation_time)
        )
    )
    logger.logo(
        "{} generated: {} exceptions {} duplicated {} crashes {} timeouts {} notarget".format(
            num_generated,
            num_exception,
            num_duplicated,
            num_crash,
            num_timeout,
            num_notarget,
        )
    )

    return (
        ga.info_code,
        ga.get_p(),
        crashes,
        generation_time,
        validation_time,
        total_run_time,
        total_programs,
    )


def generate(args, model: SpanLM):
    """
    :param args:
    :param model:
    :return:
    """
    os.makedirs(args.folder, exist_ok=True)
    os.makedirs(os.path.join(args.folder, "seed"), exist_ok=True)
    os.makedirs(os.path.join(args.folder, "valid"), exist_ok=True)
    os.makedirs(os.path.join(args.folder, "flaky"), exist_ok=True)
    os.makedirs(os.path.join(args.folder, "hangs"), exist_ok=True)
    os.makedirs(os.path.join(args.folder, "crash"), exist_ok=True)
    os.makedirs(os.path.join(args.folder, "exception"), exist_ok=True)
    os.makedirs(os.path.join(args.folder, "notarget"), exist_ok=True)
    with open(os.path.join(args.folder, "args.txt"), "w") as f:
        f.write(str(args))

    filepath = os.path.dirname(os.path.realpath(__file__))
    logger = Logger(os.path.join(filepath, args.folder))

    gen_ret = {}
    infill_ph = model.infill_ph if model is not None else "<|mask:{}|>"
    if args.library == "torch":
        apis = get_initial_programs(
            args.seedfolder, infill_ph, args.library, "argument", target_api=args.api
        )
    else:
        apis = get_initial_programs(
            args.seedfolder, infill_ph, args.library, "argument", target_api=args.api
        )

    if (args.api not in apis) and args.api != "all":
        logger.logo("Did not find {} in list of valid seed apis".format(args.api))
        return

    for api, v in apis.items():

        if args.api != api and args.api != "all":
            continue
        if len(v) == 0:
            continue
        logger.logo("--- Generating for {} ---".format(api))
        logger.logo("------ | seeds | = {} -----".format(len(apis[api])))
        seeds_for_generation = []
        for idx, seed in enumerate(apis[api]):
            status, msg = validate_status(
                seed["original"],
                args.library,
                validate_mode=args.validate_mode,
                test_executor=mp_executor.test_executor,
            )
            initial = status == ExecutionStatus.SUCCESS
            with open(
                os.path.join(args.folder, "seed", api + "_seed{}.py".format(idx + 1)),
                "w",
            ) as f:
                f.write(seed["original"])
            if initial or not args.only_valid:
                seeds_for_generation.append(seed["original"])
        logger.logo(
            "--- seeds_for_generation : {} ---".format(len(seeds_for_generation))
        )
        if len(seeds_for_generation) > 0:
            gen_ret[api] = {}
            gen_ret[api]["seeds"] = seeds_for_generation
            gen_ret[api]["initials"] = seeds_for_generation
            (
                gen_ret[api]["outputs"],
                gen_ret[api]["p"],
                gen_ret[api]["crashes"],
                gen_ret[api]["g_time"],
                gen_ret[api]["v_time"],
                gen_ret[api]["tot_time"],
                gen_ret[api]["tot_prog"],
            ) = generate_loop(
                args, model, seeds_for_generation, api, logger, args.max_valid
            )

            for idx, code in enumerate(gen_ret[api]["crashes"]):
                with open(
                    os.path.join(args.folder, api + "_crash" + str(idx + 1) + ".py"),
                    "w",
                ) as f:
                    f.write(code)

        import torch

        torch.cuda.empty_cache()
        mp_executor.test_executor.restart()

        t_start = time.time()
        with open(os.path.join(args.folder, "outputs.json"), "a") as f:
            f.write("\n")
            f.write(json.dumps(gen_ret))

    print("done")


def main():
    print("Current directory: ", os.getcwd())
    print("Results will be dumped to: ", os.path.join(os.getcwd(), "Results"))
    parser = argparse.ArgumentParser()
    # Experiment setup configs
    parser.add_argument("--model_name", type=str, default="facebook/incoder-1B")
    parser.add_argument(
        "--library", type=str, default=None, help="either 'torch' or 'tf'"
    )
    parser.add_argument("--api", type=str, default=None)
    parser.add_argument("--apilist", type=str, default=None)
    parser.add_argument("--startid", type=int, default=0)
    parser.add_argument("--endid", type=int, default=-1)
    parser.add_argument("--folder", type=str, default="Result/test")
    parser.add_argument(
        "--seedfolder", type=str, default="../codex_seed_programs/pt-codex/raw"
    )
    parser.add_argument("--use_sample_apis", action="store_true", default=False)
    parser.add_argument("--random_seed", type=int, default=420)

    # Hyperparameters
    parser.add_argument("--max_valid", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--seed_pool_size", type=int, default=30)

    # Algorithm configs
    parser.add_argument("--only_valid", action="store_true", default=False)
    parser.add_argument("--relaxargmut", action="store_true", default=False)
    parser.add_argument(
        "--seed_selection_algo",
        type=str,
        default="random",
        choices=["fitness", "random", "coverage"],
    )
    parser.add_argument(
        "--mutator_selection_algo",
        type=str,
        default="epsgreedy",
        choices=["heuristic", "epsgreedy", "ucb", "random", "ts"],
    )
    # Use a single mutator, for debug / ablation study
    parser.add_argument("--use_single_mutator", action="store_true", default=False)
    parser.add_argument("--replace_type", type=str, default=None)
    parser.add_argument(
        "--mutator_set",
        type=str,
        default="all",
        choices=["all", "noprefix", "nosuffix", "noargument", "nomethod"],
    )

    # Misc
    parser.add_argument(
        "--validate_mode",
        type=str,
        default="multiprocess",
        choices=["process", "multiprocess"],
    )
    parser.add_argument("--close_fd_mask", type=int, default=1)

    args = parser.parse_args()
    if args.library not in ["torch", "tf"]:
        raise NotImplementedError

    if args.api == "all":
        run_args = ["python"] + argparse._sys.argv
        if args.apilist is not None:
            with open(args.apilist, "r") as f:
                all_apis = f.read().splitlines()
            if args.endid != -1:
                all_apis = all_apis[: args.endid]
            all_apis = all_apis[args.startid :]
        else:
            all_apis = load_apis(args.library, args.use_sample_apis)
        ind = run_args.index("all")
        num_apis = len(all_apis)
        for api_idx, api in enumerate(all_apis):
            print("[{} / {}] {}".format(api_idx, num_apis, api))
            peek_seeds = glob.glob(os.path.join(args.seedfolder, api, "*.py"))
            if len(peek_seeds) == 0:
                print("---Skip {} for lack of valid seed---".format(api))
                continue
            if os.path.exists(
                os.path.join(args.folder, "seed", "{}_seed1.py".format(api))
            ):
                print("---Skip {} because seed1.py already exists---".format(api))
                continue
            run_args_api = run_args.copy()
            run_args_api[ind] = api
            run_cmd(run_args_api, timeout=args.timeout + 50, verbose=True)
        exit(0)

    print("> api: ", args.api)
    peek_seeds = glob.glob(os.path.join(args.seedfolder, args.api, "*.py"))
    if len(peek_seeds) == 0:
        print("---Skip {} for lack of valid seed---".format(args.api))
        exit(0)
    # avoid redundant run
    if args.api != "all" and os.path.exists(
        os.path.join(args.folder, "seed", "{}_seed1.py".format(args.api))
    ):
        print("---Skip {} because seed1.py already exists---".format(args.api))
        exit(0)
    mp_executor.init_test_executor(args, cov=(args.seed_selection_algo == "coverage"))

    model = SpanLM(args.model_name, batch_size=args.batch_size)
    set_seed(args.random_seed)
    generate(args, model)

    mp_executor.kill_executors()


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    try:
        main()
    except Exception as e:
        print(e)
    mp_executor.kill_executors()
