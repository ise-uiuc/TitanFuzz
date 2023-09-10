import ast
import glob
import json
import os
import pprint
import re

import astunparse

from util.astpasses import PassRemoveDocstring, SearchCall
from util.instrumentor import SearchAllCall, SnippetInfill
from util.util import get_unified_diff


# Remove prints and imports
# prints -> lots of syntax errors
# imports, froms needs to think about a bit more
def remove_prints_and_imports(original: str) -> str:
    lines = original.splitlines()
    c_lines = [
        x
        for x in lines
        if not x.startswith("print")
        and not x.startswith("import")
        and not x.startswith("from")
    ]
    return "\n".join(c_lines)


def remove_comments(original: str) -> str:
    lines = original.splitlines()
    c_lines = [
        x for x in lines if not x.rstrip().startswith("#") and not x.strip() == ""
    ]
    code = "\n".join(c_lines)
    try:
        root = ast.parse(code)
        PassRemoveDocstring().remove_docstring(root)
        modified = ast.fix_missing_locations(root)
        code_cleaned = astunparse.unparse(root)
    except:
        return code

    return code_cleaned


def remove_empty_line(original: str) -> str:
    lines = original.splitlines()
    c_lines = [x for x in lines if not x.strip() == ""]
    return "\n".join(c_lines)


def remove_main(original: str) -> str:
    code = original.replace('if __name__ == "__main__":', "if True:")
    code = code.replace("if __name__ == '__main__':", "if True:")
    return code


tag_pattern = re.compile(r"<.+>")  # such as <cell>, <test>, </cell>


def remove_tags(code: str) -> str:
    clean_code = re.split(tag_pattern, code)[0]
    return clean_code


def remove_skip_api(code: str) -> str:
    # If not removed, raise FATAL Flags parsing error: Unknown command line flag 'library'
    bad_codes = [
        "tf.test.main()",
        "tf.compat.v1.test.main()",
        "disable_eager_execution()",
        "disable_v2_behavior",
        "InteractiveSession",
        "exit()",
    ]
    for bad_code in bad_codes:
        code = code.replace(bad_code, "")
    return code


def remove_seed_setting(code: str) -> str:
    return re.sub("torch\.manual_seed\(\S+\)", "", code)


def clean_raw_code(original: str) -> str:
    code = remove_prints_and_imports(original)
    code = remove_tags(code)
    return code


def syntax_fix_remove_last_line(original):
    """Remove last line until syntax pass."""
    code = original
    while len(code) > 0:
        syntax_error = False
        try:
            o_ast = ast.parse(code)
            node = ast.fix_missing_locations(o_ast)
            code = astunparse.unparse(node).strip()
            node = ast.parse(code)  # reparse
        except Exception as e:  # if the snippet is not valid python Syntax
            syntax_error = True
        if syntax_error:
            code = "\n".join(code.splitlines()[:-1])
        else:
            return code
    return code


session_pattern1 = re.compile(r"with tf.compat.v1.Session()")
session_pattern2 = re.compile(r"with tf.Session()")


def remove_session(original):
    code = re.split(session_pattern1, original)[0]
    code = re.split(session_pattern2, code)[0]
    return code


def remove_cuda(original):
    lines = original.splitlines()
    c_lines = [x for x in lines if "cuda" not in x and not x.startswith("from")]
    return "\n".join(c_lines)


def remove_name_keywords(snippet: str):
    try:
        root = ast.parse(snippet)
        call_nodes = SearchAllCall().search_from_ast(root)
        for node_name in call_nodes:
            node, call_name = node_name
            if "keywords" in dir(
                node
            ):  # all calls should have keyword arguments already
                node.keywords = [a for a in node.keywords if a.arg != "name"]

        modified = ast.fix_missing_locations(root)
        code_cleaned = astunparse.unparse(modified)
        return code_cleaned
    except Exception as e:
        return snippet


def clean_code(
    code: str,
    prints_and_imports=False,
    comment=False,
    cuda=False,
    fix_syntax=True,
    fix_session=True,
    remove_func=True,
) -> str:
    if remove_func:
        # do not consider code with has function declarations
        if re.search(r"def\s+\S+\(.*\)\:", code):
            return ""

    # Fix with regular expression
    code = remove_tags(code)
    code = remove_skip_api(code)
    code = remove_seed_setting(code)
    code = remove_empty_line(code)
    code = remove_main(code)

    if prints_and_imports:
        code = remove_prints_and_imports(code)
    if comment:
        code = remove_comments(code)
    if cuda:
        code = remove_cuda(code)
    # Fix syntax error
    if fix_syntax:
        code = syntax_fix_remove_last_line(code)

    code = remove_name_keywords(code)

    if fix_session:
        code = remove_session(code)

    return code


def get_initial_programs(
    directory: str,
    mask_identifier: str,
    library: str,
    replace_type: str,
    target_api="all",
) -> dict:
    """
    Get all initial programs from the directory.
    structure:
        directory |
            api_1
            api_2
    """
    ret = {}
    syntax_error, multi_api_calls, no_api_call, successful = 0, 0, 0, 0

    apis = glob.glob(os.path.join(directory, "*"))
    if target_api != "all":
        apis = [os.path.join(directory, target_api)]
    if library == "tf":
        with open("data/tf_apis.txt", "r") as f:
            tf_apis = f.read().splitlines()
        tf_apis = [os.path.join(directory, a) for a in tf_apis]
        apis = list(set(tf_apis).intersection(apis))

    for api in apis:
        api_name = api.split("/")[-1]
        ret[api_name] = []
        for program in glob.glob(os.path.join(api, "*.py")):
            with open(program, "r") as f:
                api_call = api.split("/")[-1].split(".")[-1]
                # Keep the programs with functions because they help to cover APIs.
                original = clean_code(
                    f.read(), prints_and_imports=True, comment=True, cuda=True
                )
                infill = SnippetInfill(
                    mask_identifier=mask_identifier,
                    api_call=api_call,
                    prefix=".".join(api.split("/")[-1].split(".")[1:-1]),
                    library=library,
                    replace_type=replace_type,
                )
                num_replaced, infill_code, original_code = infill.add_infill(original)
                if num_replaced == -1:
                    syntax_error += 1
                elif num_replaced >= 1:  # start with single or multi api call
                    # print(get_unified_diff(original_code, infill_code))
                    successful += 1
                    ret[api.split("/")[-1]].append(
                        {"original": original_code, "infill": infill_code}
                    )
                else:
                    no_api_call += 1
        if len(ret[api_name]) == 0:
            del ret[api_name]
    print(
        "Syntax error: {} | Multi-API calls: {} | No Api calls: {} | Successful: {}".format(
            syntax_error, multi_api_calls, no_api_call, successful
        )
    )
    return ret


def get_initial_seed_programs(directory: str, library: str, args) -> list:
    """
    Get all initial programs from the directory.
    structure:
        directory |
            api_1 |
                1.py
                2.py
                ...
            api_2
    Returns a list of [api, label, original_code]
    """

    if os.path.exists(os.path.join(directory, "outputs.json")):
        with open(os.path.join(directory, "outputs.json"), "r") as f:
            genlog = json.load(f)
        gen_time = 0
        for api, records in genlog.items():
            gen_time += records["g_time"]
        print(
            "Generation time: {} s for a total of {} APIs".format(gen_time, len(genlog))
        )
        with open(os.path.join(args.output_dir, "process.log"), "a") as f:
            f.write(
                "Generation time: {} s for a total of {} APIs\n".format(
                    gen_time, len(genlog)
                )
            )

    tasks = []
    programs = glob.glob(os.path.join(directory, "*.py"))
    if len(programs) == 0:

        for api in glob.glob(os.path.join(directory, "*")):
            for program in glob.glob(os.path.join(api, "*.py")):
                with open(program, "r") as f:
                    api = api.split("/")[-1]
                    label = api + "_" + program.split("/")[-1].split(".")[0]
                    original_code = f.read()
                    if args.api is not None and (api != args.api):
                        continue
                    if args.id is not None and (label != api + "_" + str(args.id)):
                        continue
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
            if "seed" in label:
                continue
            with open(program, "r") as f:
                original_code = f.read()
            tasks.append([api, label, original_code])
    return tasks


def clean_programs(tasks, args) -> dict:
    """
    Cleans programs via various static techniques.

    Input is a list of tasks [api, label, original_code]
    Returns a dictionary of { api:
        [ {"label": label, "original": original_code, "fix": fixed_code} ] }
    """
    ret = dict()
    syntax_error, no_api_call, api_call_successful = 0, 0, 0
    for api, label, original_code in tasks:
        # Note that in validate seed we keep functions.
        code = clean_code(
            original_code,
            prints_and_imports=True,
            comment=True,
            cuda=True,
            remove_func=False,
        )
        try:
            o_ast = ast.parse(code)
            code = astunparse.unparse(o_ast).strip()
        except Exception as e:  # if the snippet is not valid python syntax
            syntax_error += 1
        else:
            # Disable static check for no-call for now.
            api_call_successful += 1
            if api not in ret:
                ret[api] = []
            ret[api].append({"label": label, "original": original_code, "fix": code})

    print(
        "Syntax error: {} | No Api calls: {} | API Successful: {}".format(
            syntax_error, no_api_call, api_call_successful
        )
    )
    with open(os.path.join(args.output_dir, "process.log"), "a") as f:
        f.write(
            "Syntax error: {} | No Api calls: {} | API Successful: {}\n".format(
                syntax_error, no_api_call, api_call_successful
            )
        )
    with open(os.path.join(args.output_dir, "fix.json"), "w") as f:
        json.dump(ret, f)
    return ret
