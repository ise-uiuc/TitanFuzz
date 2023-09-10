# perform dead code elimination
# suppose chain structure
import ast

import astunparse
import numpy as np

from util.astpasses import PassRemoveDocstring, SearchCall
from util.instrumentor import UseDef


def _dfs(u, e, n, visited):
    for v in range(1, n):
        if (e[u, v] == 1) and (v not in visited):
            visited.add(v)
            visited = _dfs(v, e, n, visited)
    return visited


def _dead_code_elim(target_line, ids, defs, uses):
    """Eliminate dead code and returns the useful line numbers.

    Here, dead code is defined as not reachable wrt the target line.
    """

    linenos = set(defs.keys()).union(set(uses.keys()))
    linenos = list(linenos)
    linenos.sort()
    n = max(linenos) + 1

    # Build reachability graph
    edge = np.zeros((n, n), np.int32)
    edge_rev = np.zeros((n, n), np.int32)
    live_id_defs = {}

    keep_lines = set()
    for i in range(1, n):
        if i in uses:
            for id in uses[i]:
                if id not in live_id_defs:
                    continue
                j = live_id_defs[id]
                edge[j, i] = 1
                edge_rev[i, j] = 1
        if i in defs:
            for id in defs[i]:
                live_id_defs[id] = i
        # Keep these lines, probably control flows.
        if (i not in uses) and (i not in defs):
            keep_lines.add(i)

    # Find the successor
    successors = _dfs(target_line, edge, n, {target_line})

    successors = list(range(min(successors), max(successors) + 1))

    # Find the predecessor
    predecessors = set()
    for successor in successors:
        predecessors_ = _dfs(successor, edge_rev, n, {successor})
        predecessors.update(predecessors_)

    return set(keep_lines).union(set(predecessors).union(set(successors)))


def dead_code_elim(code, api, verbose=False) -> str:
    lib_prefix = api.split(".")[0] + "."
    api_call = api.split(".")[-1]
    prefix = api.lstrip(lib_prefix)[: -len(api_call)].rstrip(".")

    try:
        o_ast = ast.parse(code)
        original_code = astunparse.unparse(o_ast).strip()
        o_ast = ast.parse(original_code)  # reparse
    except Exception as e:  # if the snippet is not valid python syntax
        # print(e)
        # print("Error parsing snippet")
        return code

    ids, defs, uses = UseDef(lib_prefix=lib_prefix).get_use_def(original_code)
    if verbose:
        print("lib_prefix: ", lib_prefix)
        print("api_call: ", api_call)
        print("prefix:", prefix)
        print("ids: ", ids)
        print("defs: ", defs)
        print("uses: ", uses)

    # Search for the target API call
    _, nodes = SearchCall(api_call=api_call, prefix=prefix).search(o_ast)
    if len(nodes) == 0:
        # print("No target call")
        return original_code

    # Take the first call of target API as the target line
    target_line = min([n.lineno for n in nodes])

    if verbose:
        print("target_line: ", target_line)
    n = nodes[0]

    # Elminate the code that cannot reach target line or cannot be reached from target line
    keep_lines = _dead_code_elim(target_line, ids, defs, uses)

    # Moderate elimination: Keep a consequtive code block to avoid issues
    # line_start, line_end = min(keep_lines), max(keep_lines)
    # keep_lines = list(range(line_start-1, line_end))

    # Radical elimination: Keep only dependent lines
    lines = original_code.splitlines()

    # Keep indented code
    for idx, line in enumerate(lines):
        if line.startswith(" "):
            keep_lines.add(idx)
    keep_lines = list(keep_lines)
    keep_lines.sort()

    lines = [lines[idx - 1] for idx in keep_lines]

    code_cleaned = "\n".join(lines)

    # Remove prompts that become a constant string line
    try:
        root = ast.parse(code_cleaned)
    except:
        code_cleaned = original_code
        root = ast.parse(code_cleaned)

    try:
        root = ast.parse(code_cleaned)
        PassRemoveDocstring().remove_docstring(root)
        modified = ast.fix_missing_locations(root)
        code_cleaned = astunparse.unparse(root)
    except:
        pass

    return code_cleaned


def test():
    api = "torch.nn.ConstantPad2d"
    with open("util/example.py", "r") as f:
        code = f.read()
    code_cleaned = dead_code_elim(code, api, verbose=False)
    print(code_cleaned)
