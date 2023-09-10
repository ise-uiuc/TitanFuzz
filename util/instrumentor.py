import ast
import random

import astunparse

from util import util


class MultiKeywordFinder(ast.NodeTransformer):
    def __init__(self, lib_prefix: str):
        self.ignore_keywords = ["dtype", "device", "requires_grad", "name"]
        self.lib_prefix = lib_prefix
        self.keyword_calls = 0

    def count(self, snippet: str):
        o_ast = ast.parse(snippet)
        self.visit(o_ast)
        return self.keyword_calls

    def visit_Call(self, node: ast.Call):
        if "attr" in dir(node.func):
            temp_node = node.func
            api_call = ""
            while "value" in dir(temp_node) and "attr" in dir(temp_node):
                api_call = temp_node.attr + "." + api_call
                temp_node = temp_node.value
            if "id" in dir(temp_node):
                api_call = temp_node.id + "." + api_call
            if api_call.endswith("."):
                api_call = api_call[:-1]
            if api_call.startswith(self.lib_prefix):
                if "keywords" in dir(node):
                    for k in node.keywords:
                        if "arg" in dir(k) and k.arg not in self.ignore_keywords:
                            self.keyword_calls += 1
            self.generic_visit(node)
            return node

    def visit(self, node):
        if isinstance(node, ast.Call):
            return self.visit_Call(node)
        self.generic_visit(node)
        return node


class DepthFinder(ast.NodeTransformer):
    def __init__(self, lib_prefix: str):
        self.current_var = None
        self.current_depth = None
        self.api_depth = 0
        self.seen = False
        self.current_line = -1
        self.variables = {}  # handles current variable depth
        self.variables_max = {}  # handles maximum variable depth (in case of rewrites)
        self.lib_prefix = lib_prefix

    def visit_Call(self, node: ast.Call):
        if "attr" in dir(node.func):
            temp_node = node.func
            api_call = ""
            while "value" in dir(temp_node) and "attr" in dir(temp_node):
                api_call = temp_node.attr + "." + api_call
                temp_node = temp_node.value
            if "id" in dir(temp_node):
                api_call = temp_node.id + "." + api_call
            if api_call.endswith("."):
                api_call = api_call[:-1]
            if api_call.startswith(self.lib_prefix):
                temp_api_depth = self.api_depth
                self.api_depth += 1
            if self.current_var is not None:
                self.current_depth = max(self.current_depth, self.api_depth)
            self.generic_visit(node)
            if api_call.startswith(self.lib_prefix):
                self.api_depth = temp_api_depth
            return node
        else:
            self.generic_visit(node)
            return node

    def visit_Name(self, node: ast.Name):
        if node.id == self.current_var and node.id in self.variables:
            if self.seen:
                self.current_depth = max(
                    self.current_depth, self.variables[node.id] + self.api_depth
                )
            else:  # swallow the first occurrence (left side)
                self.seen = True
        elif self.current_var is not None and node.id in self.variables:
            self.current_depth = max(
                self.current_depth, self.variables[node.id] + self.api_depth
            )
        self.generic_visit(node)
        return node

    def visit_Assign(self, node: ast.Assign):
        # just use first target for now + handles reusing variable name
        if "id" in dir(node.targets[0]):
            self.current_var = node.targets[0].id
            self.current_depth = 0
        self.generic_visit(node)
        return node

    def max_depth(self, snippet: str):
        try:
            o_ast = ast.parse(snippet)
            original_code = astunparse.unparse(o_ast).strip()
            o_ast = ast.parse(original_code)  # reparse
        except:  # if the snippet is not valid python syntax
            # print("Error parsing snippet")
            return -1, "", ""

        self.visit(o_ast)
        if len(self.variables_max) == 0:
            return 0
        return max([v for _, v in self.variables_max.items()])

    def visit(self, node):
        # reset
        if "lineno" in dir(node) and node.lineno != self.current_line:
            if self.current_var is not None:
                self.variables[self.current_var] = self.current_depth
                if self.current_var not in self.variables_max:
                    self.variables_max[self.current_var] = self.current_depth
                else:
                    self.variables_max[self.current_var] = max(
                        self.variables_max[self.current_var], self.current_depth
                    )
            self.current_var = None
            self.seen = False
            self.current_line = node.lineno
            self.api_depth = 0

        if isinstance(node, ast.Assign):
            return self.visit_Assign(node)

        if isinstance(node, ast.Call):
            return self.visit_Call(node)

        if isinstance(node, ast.Name):
            return self.visit_Name(node)

        self.generic_visit(node)
        return node


def get_api_call_from_node(node):
    temp_node = node.func
    api_call = ""
    while "value" in dir(temp_node) and "attr" in dir(temp_node):
        api_call = temp_node.attr + "." + api_call
        temp_node = temp_node.value
    if "id" in dir(temp_node):
        api_call = temp_node.id + "." + api_call
    if api_call.endswith("."):
        api_call = api_call[:-1]
    return api_call


class SearchAllCall(ast.NodeTransformer):
    """Search an specific API call."""

    def __init__(self):
        self.call_nodes = []

    def search_from_ast(self, o_ast) -> list:
        self.call_nodes = []
        modified = self.visit(o_ast)
        return self.call_nodes

    def search_from_code(self, snippet) -> list:
        self.call_nodes = []
        o_ast = ast.parse(snippet)
        modified = self.visit(o_ast)
        return self.call_nodes

    def visit_Call(self, node: ast.Call):
        if "attr" in dir(node.func):
            api_call = get_api_call_from_node(node)
            self.call_nodes.append((node, api_call))
            self.generic_visit(node)
            return node
        else:
            self.generic_visit(node)
            return node

    def visit(self, node):
        if isinstance(node, ast.Call):
            return self.visit_Call(node)
        self.generic_visit(node)
        return node


class SearchAllLibCall(SearchAllCall):
    ignore_calls = [
        "tf.print",
        "tf.constant",
        "tf.zeros",
        "tf.ones" "tf.shape",
    ]

    def __init__(self, lib_prefix: str):
        super().__init__()
        self.lib_prefix = lib_prefix

    def check_if_ignore(self, api_call) -> bool:
        return api_call in self.ignore_calls or util.if_skip_api(
            api_call, self.lib_prefix
        )

    def search_from_ast(self, o_ast) -> list:
        nodes = super().search_from_ast(o_ast)
        lib_calls = [
            (node, name)
            for node, name in nodes
            if name.startswith(self.lib_prefix) and not self.check_if_ignore(name)
        ]
        return lib_calls

    def search_from_code(self, snippet) -> list:
        nodes = super().search_from_code(snippet)
        lib_calls = [
            (node, name) for node, name in nodes if name.startswith(self.lib_prefix)
        ]
        return lib_calls


class UniqueFinder(SearchAllLibCall):
    def __init__(self, lib_prefix: str):
        super().__init__(lib_prefix)
        self.unique_found_apis = {}
        self.unique_found_call_exps = {}
        self.num_calls = 0

    def count(self, snippet: str) -> (int, int, int):
        """Returns number of unique apis, exact repeated apis, and repeated apis."""
        lib_calls = self.search_from_code(snippet)
        for node, api_call in lib_calls:
            call_exp = astunparse.unparse(node).strip()
            if call_exp not in self.unique_found_call_exps:
                self.unique_found_call_exps[call_exp] = 0
            if api_call not in self.unique_found_apis:
                self.unique_found_apis[api_call] = 0
            self.unique_found_apis[api_call] += 1
            self.unique_found_call_exps[call_exp] += 1
        return (
            len(self.unique_found_apis),
            sum([v - 1 for _, v in self.unique_found_call_exps.items()]),
            sum([v - 1 for _, v in self.unique_found_apis.items()]),
        )


# Basic Infilling parser to replace the original code with mask_identifier for infilling
# Note that doing this will remove newlines and other formatting (including comments)
# Adopted from FASER project
class SnippetInfill(ast.NodeTransformer):
    def __init__(
        self,
        mask_identifier: str,
        api_call: str,
        prefix: str,
        library: str,
        replace_type: str = "argument",
    ):
        self.mask_identifier = mask_identifier
        self.api_call = api_call
        self.num_replaced = 0
        self.line_no = -1
        self.prefix = prefix
        self.library = library
        self.replace_type = replace_type
        self.replace = False

    def add_infill(self, snippet: str) -> (int, str, str):
        # instrument base class if present: imports, print, sample_to_str (only if different file)
        try:
            o_ast = ast.parse(snippet)
            original_code = astunparse.unparse(o_ast).strip()
            o_ast = ast.parse(original_code)  # reparse
        except:  # if the snippet is not valid python syntax
            # print("Error parsing snippet")
            return -1, "", ""
        self.visit(o_ast)
        if self.num_replaced < 1:
            return self.num_replaced, "", original_code
        self.replace, self.num_replaced = True, 0
        modified = self.visit(o_ast)
        modified = ast.fix_missing_locations(modified)
        if self.replace_type == "argument":
            infill_code = (
                astunparse.unparse(modified)
                .strip()
                .replace(
                    "'{}'".format(self.mask_identifier), self.mask_identifier.format(0)
                )
            )
        elif self.replace_type == "prefix":
            infill_code = (
                "import torch\n"
                if self.library == "torch"
                else "import tensorflow as tf\n"
            )
            infill_code += "import numpy as np\n"
            end_replace = random.randint(0, self.line_no - 1)
            start_replace = random.randint(0, end_replace)
            infill_code += "\n".join(original_code.splitlines()[:start_replace])
            if start_replace != 0:
                infill_code += "\n"
            infill_code += (
                self.mask_identifier.format(0)
                + "\n"
                + "\n".join(original_code.splitlines()[end_replace:])
            )
        elif self.replace_type == "suffix":
            start_replace = random.randint(
                self.line_no, len(original_code.splitlines())
            )
            end_replace = random.randint(start_replace, len(original_code.splitlines()))
            infill_code = (
                "\n".join(original_code.splitlines()[:start_replace])
                + "\n"
                + self.mask_identifier.format(0)
            )
            if end_replace != len(original_code.splitlines()):
                infill_code += "\n"
            infill_code += "\n".join(original_code.splitlines()[end_replace:])
        elif self.replace_type == "prefix-argument":
            t_infill_code = (
                astunparse.unparse(modified)
                .strip()
                .replace(
                    "'{}'".format(self.mask_identifier), self.mask_identifier.format(1)
                )
            )
            infill_code = (
                "import torch\n"
                if self.library == "torch"
                else "import tensorflow as tf\n"
            )
            infill_code += "import numpy as np\n"
            end_replace = random.randint(0, self.line_no - 1)
            start_replace = random.randint(0, end_replace)
            infill_code += "\n".join(t_infill_code.splitlines()[:start_replace])
            if start_replace != 0:
                infill_code += "\n"
            infill_code += (
                self.mask_identifier.format(0)
                + "\n"
                + "\n".join(t_infill_code.splitlines()[end_replace:])
            )
        elif self.replace_type == "suffix-argument":
            t_infill_code = (
                astunparse.unparse(modified)
                .strip()
                .replace(
                    "'{}'".format(self.mask_identifier), self.mask_identifier.format(0)
                )
            )
            start_replace = random.randint(
                self.line_no, len(t_infill_code.splitlines())
            )
            end_replace = random.randint(start_replace, len(t_infill_code.splitlines()))
            infill_code = (
                "\n".join(t_infill_code.splitlines()[:start_replace])
                + "\n"
                + self.mask_identifier.format(1)
            )
            if end_replace != len(t_infill_code.splitlines()):
                infill_code += "\n"
            infill_code += "\n".join(t_infill_code.splitlines()[end_replace:])
        else:
            assert False

        return self.num_replaced, infill_code, original_code

    def visit_Call(self, node: ast.Call):
        if "attr" in dir(node.func) and self.api_call == node.func.attr:
            temp_node = node.func
            prefix = ""
            while "value" in dir(temp_node) and "attr" in dir(temp_node.value):
                prefix = temp_node.value.attr + "." + prefix
                temp_node = temp_node.value
            if prefix.endswith("."):
                prefix = prefix[:-1]
            if prefix != self.prefix:
                self.generic_visit(node)
                return node
            self.num_replaced += 1
            if not self.replace:
                self.generic_visit(node)
                return node
            # this will give 'mask_identifier' as the argument for the function
            # TODO: extend this to support picking which arguments
            if "argument" in self.replace_type:
                node.args = [ast.Constant(value=self.mask_identifier)]
                if "keywords" in dir(node):
                    node.keywords = []
                self.replace = False  # do not replace any later api calls
            self.line_no = node.lineno
            self.generic_visit(node)
            return node
        else:
            self.generic_visit(node)
            return node

    def visit(self, node):
        if isinstance(node, ast.Call):
            return self.visit_Call(node)
        self.generic_visit(node)
        return node


class SnippetInfillArbitratyAPI(ast.NodeTransformer):
    def __init__(
        self,
        mask_identifier: str,
        api_call_list: list,
        full_api_list: list,
        lib_prefix: str = "torch",
    ):
        self.mask_identifier = mask_identifier
        self.api_call_list = api_call_list
        self.full_api_list = full_api_list
        self.lib_prefix = lib_prefix
        self.num_replaced = 0

    def add_infill(self, node, add_keywords=False, replace_method=False):
        self.num_replaced = 0
        self.call_nodes = []
        self.line_nos = []
        try:
            original_code = astunparse.unparse(node).strip()
        except:  # if the snippet is not valid python syntax
            print("Error parsing snippet")
            return -1, "", ""
        self.visit(node)
        n_call = len(self.call_nodes)
        if n_call > 0:
            replace_node = self.call_nodes[random.randint(0, n_call - 1)]
            if not add_keywords:
                if replace_method:
                    replace_node.func = ast.Attribute(
                        value=ast.Name(id=self.lib_prefix),
                        attr=self.mask_identifier.format(0),
                    )
                else:
                    replace_node.args = [
                        ast.Constant(value=self.mask_identifier.format(0))
                    ]
                    if "keywords" in dir(replace_node):
                        replace_node.keywords = []
            else:
                if "keywords" in dir(
                    replace_node
                ):  # all calls should have keyword arguments already
                    replace_node.keywords.append(
                        ast.keyword(
                            arg=self.mask_identifier.format(0),
                            value=ast.Constant(value=self.mask_identifier.format(1)),
                        )
                    )
            self.num_replaced += 1
        modified = ast.fix_missing_locations(node)
        if not add_keywords:
            infill_code = (
                astunparse.unparse(modified)
                .strip()
                .replace(
                    "'{}'".format(self.mask_identifier.format(0)),
                    self.mask_identifier.format(0),
                )
            )
        else:
            infill_code = (
                astunparse.unparse(modified)
                .strip()
                .replace(
                    "'{}'".format(self.mask_identifier.format(1)),
                    self.mask_identifier.format(1),
                )
            )
        return self.num_replaced, infill_code, original_code

    def visit_Call(self, node: ast.Call):
        if "attr" in dir(node.func):
            temp_node = node.func
            api_call = ""
            while "value" in dir(temp_node) and "attr" in dir(temp_node):
                api_call = temp_node.attr + "." + api_call
                temp_node = temp_node.value
            if "id" in dir(temp_node):
                api_call = temp_node.id + "." + api_call
            if api_call.endswith("."):
                api_call = api_call[:-1]
            if api_call not in self.full_api_list:
                self.generic_visit(node)
                return node
            self.call_nodes.append(node)
            self.line_nos.append(node.lineno)
            self.generic_visit(node)
            return node
        else:
            self.generic_visit(node)
            return node

    def visit(self, node):
        if isinstance(node, ast.Call):
            return self.visit_Call(node)
        self.generic_visit(node)
        return node


class UseDef(ast.NodeTransformer):
    """Simple use-def analysis, assumes DFG is a DAG."""

    def __init__(self, lib_prefix: str):
        self.uses = {}
        self.defs = {}
        self.ids = {}
        self.lib_prefix = lib_prefix

    def visit_Name(self, node: ast.Name):
        lineno = node.lineno
        id = node.id
        ctx = node.ctx
        if isinstance(ctx, ast.Load):
            if lineno not in self.uses:
                self.uses[lineno] = set()
            self.uses[lineno].add(id)
        if isinstance(ctx, ast.Store):
            if lineno not in self.defs:
                self.defs[lineno] = set()
            self.defs[lineno].add(id)

        # self.generic_visit(node)
        return node

    def get_use_def(self, snippet: str):
        try:
            o_ast = ast.parse(snippet)
            original_code = astunparse.unparse(o_ast).strip()
            o_ast = ast.parse(original_code)  # reparse
        except:  # if the snippet is not valid python syntax
            # print("Error parsing snippet")
            return -1, "", ""

        self.visit(o_ast)
        return self.ids, self.defs, self.uses

    def visit(self, node):
        if isinstance(node, ast.Name):
            return self.visit_Name(node)
        self.generic_visit(node)


if __name__ == "__main__":
    code = """input = torch.randn(1, 3, requires_grad=True)
input = torch.clamp(input, min=0.0, max=1.0)"""
    print(UniqueFinder("torch").count(code))
    code = """input = torch.randn(1, 3, requires_grad=True)
input = torch.clamp(input, min=0.0, max=1.0)
x = torch.exp(input, min=0.0, max=1.0)
x = torch.exp(x, min=0.0, max=1.0)
y = torch.exp(x, min=0.0, max=1.0)
input_tensor = input.clone().set(1)
x = np.random(torch.clone(input))"""
    print(UniqueFinder("torch").count(code))
    print(DepthFinder("torch").max_depth(code))
    print(SearchAllCall().search_from_code(code))
