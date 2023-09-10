import ast

import astunparse


class NodeTransformerWithPrePost(ast.NodeTransformer):
    def visit(self, node):
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        node.pre = []
        node.post = []
        return visitor(node)

    def generic_visit(self, node: ast.AST) -> ast.AST:
        for field, old_value in ast.iter_fields(node):
            if not hasattr(node, "pre"):
                node.pre = []
                node.post = []
            if isinstance(old_value, list):
                isScope = field == "body"
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    if hasattr(value, "pre"):
                        if isScope:  # insert pres if this is a scope
                            new_values.extend(value.pre)
                        else:  # leave for outer scope if this is not one
                            node.pre.extend(value.pre)
                    new_values.append(value)
                    if hasattr(value, "post"):
                        if isScope:  # insert posts if this is a scope
                            new_values.extend(value.post)
                        else:  # leave for outer scope if this is not one
                            node.pre.extend(value.post)
                old_value[:] = new_values
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
                    if hasattr(new_node, "pre"):
                        node.pre.extend(new_node.pre)
                        node.post.extend(new_node.post)
        return node

    def reset(self) -> None:
        pass


class PassFlattenCall(NodeTransformerWithPrePost):
    # make every call directly assigned to a variable

    __tempId = 0

    def __init__(self) -> None:
        # any call not directly assigned is nested
        self.nestedCall = False

    def reset(self) -> None:
        self.__tempId = 0

    def newTempVar(self) -> ast.Name:
        self.__tempId += 1
        return ast.Name(id="PassFlattenCallTempVar{}".format(self.__tempId))

    # rewrite generic_visit() for statement prepending and nestedCall tagging
    def generic_visit(self, node: ast.AST) -> ast.AST:
        isAssign = isinstance(node, ast.Assign)
        if not isAssign:  # set nestedCall mark for anything but assign
            oldNestedCall0 = self.nestedCall
            self.nestedCall = True
        for field, old_value in ast.iter_fields(node):
            if isinstance(old_value, list):
                isScope = field == "body"
                if isScope:  # clear nestedCall mark for new scope
                    oldNestedCall1 = self.nestedCall
                    self.nestedCall = False
                new_values = []
                for value in old_value:
                    if isinstance(value, ast.AST):
                        value = self.visit(value)
                        if value is None:
                            continue
                        elif not isinstance(value, ast.AST):
                            new_values.extend(value)
                            continue
                    if isScope:  # insert pres if this is a scope
                        new_values.extend(value.pre)
                    else:  # leave for outer scope if this is not one
                        node.pre.extend(value.pre)
                    new_values.append(value)
                old_value[:] = new_values
                if isScope:  # restore nestedCall mark
                    self.nestedCall = oldNestedCall1
            elif isinstance(old_value, ast.AST):
                new_node = self.visit(old_value)
                if new_node is None:
                    delattr(node, field)
                else:
                    setattr(node, field, new_node)
                    node.pre.extend(new_node.pre)
        if not isAssign:  # restore nestedCall mark
            self.nestedCall = oldNestedCall0
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        oldNestedCall = self.nestedCall
        self.generic_visit(node)
        self.nestedCall = oldNestedCall  # restore nestedCall mark

        if self.nestedCall:  # Flatten this call
            nodeName = self.newTempVar()
            nodeAssign = ast.Assign(targets=[nodeName], value=node)
            nodeName.pre = node.pre
            nodeName.pre.append(nodeAssign)
            return nodeName

        return node

    def visit_ListComp(
        self, node: ast.ListComp
    ) -> ast.ListComp:  # leave alone [f(x) for x in xs]
        return node

    def visit_Lambda(self, node: ast.Lambda) -> ast.Lambda:
        return node


class PassRemoveDocstring(NodeTransformerWithPrePost):
    """Remove the docstrings from code"""

    def remove_docstring(self, node: ast.AST):
        body = node.body
        body_ = []
        for x in body:
            if (
                isinstance(x, ast.Expr)
                and isinstance(x.value, ast.Constant)
                and isinstance(x.value.value, str)
            ):
                # Delete a constant string node
                continue
            body_.append(x)
        setattr(node, "body", body_)


class PassLogTorchIntermediate(NodeTransformerWithPrePost):
    # make every call directly assigned to a variable

    __tempId = 0

    def __init__(self) -> None:
        # any call not directly assigned is nested
        self.nestedCall = False

    def reset(self) -> None:
        self.__tempId = 0

    def newTempVar(self, lval: str) -> str:
        self.__tempId += 1
        return "PassLogTorchIntermediateTempVar{}_{}".format(self.__tempId, lval)

    @staticmethod
    def getSinppet(lval: str, rval: str) -> ast.AST:
        return ast.parse(
            """
if "torch.Tensor" in str(type({})):
    {} = {}.clone()""".format(
                rval, lval, rval
            )
        ).body[0]

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        for lval in node.targets:
            if not isinstance(lval, ast.Name):  # log only names, exclude indirections
                continue
            name: str = astunparse.unparse(lval)
            name = name.strip("\n")
            node.post.append(self.getSinppet(self.newTempVar(name), name))
        return node


class PassLogTfIntermediate(NodeTransformerWithPrePost):
    # make every call directly assigned to a variable

    __tempId = 0

    def __init__(self) -> None:
        # any call not directly assigned is nested
        self.nestedCall = False

    def reset(self) -> None:
        self.__tempId = 0

    def newTempVar(self, lval: str) -> str:
        self.__tempId += 1
        return "PassLogTfIntermediateTempVar{}_{}".format(self.__tempId, lval)

    @staticmethod
    def getSinppet(lval: str, rval: str) -> ast.AST:
        return ast.parse(
            """
if "Tensor" in str(type({})):
    {} = tf.identity({})""".format(
                rval, lval, rval
            )
        ).body[0]

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        for lval in node.targets:
            if isinstance(lval, ast.Tuple):
                for elem in lval.elts:
                    if not isinstance(
                        elem, ast.Name
                    ):  # log only names, exclude indirections
                        continue
                    name: str = astunparse.unparse(elem)
                    name = name.strip("\n")
                    node.post.append(self.getSinppet(self.newTempVar(name), name))
            else:
                if not isinstance(
                    lval, ast.Name
                ):  # log only names, exclude indirections
                    continue
                name: str = astunparse.unparse(lval)
                name = name.strip("\n")
                node.post.append(self.getSinppet(self.newTempVar(name), name))
        return node


class PassReplaceAny(NodeTransformerWithPrePost):
    def __init__(self, targets: dict) -> None:
        self.targets = targets

    def visit(self, node) -> ast.AST:
        node.pre = []
        node.post = []
        node = self.generic_visit(node)
        try:
            old: str = astunparse.unparse(node).strip("\n")
            new: str = self.targets[old]
            if new != None:
                node = ast.parse(new).body[0].value
            return node
        except Exception:
            return node


class PassReplaceCallsGeneral(NodeTransformerWithPrePost):
    def __init__(self, funcs: list) -> None:
        """Requires list[func[(oldCall: str, oldArgs: str) -> (newPattern: str, newArgs: str)]]
        If newPattern is "before().new({}).after()", then replace "old(1, 2)" with "before().new(newArgs).after()"""
        self.funcs = funcs

    def visit_Call(self, node: ast.Call) -> ast.AST:
        for func in self.funcs:
            oldCall: str = astunparse.unparse(node.func).strip("\n")
            oldWhole: str = astunparse.unparse(node).strip("\n")  # "old(1, 2)"
            oldArgs: str = oldWhole[len(oldCall) + 1 : -1]  # "1, 2"
            changed, newPattern, newArgs = func(oldCall, oldArgs)
            if not changed:
                continue
            newWhole: str = newPattern.format(newArgs)  # "before().new(1, 2).after()"
            newNode: ast.AST = ast.parse(newWhole).body[0].value
            node = newNode
            break
        self.generic_visit(node)
        return node


class PassReplaceCallsIfArgsWithoutList(NodeTransformerWithPrePost):
    # If pattern is an empty string, then replace the call with string constant "old() removed"
    # If pattern is "before().new({}).after()", then replace "old(1, 2)" with "before().new(1, 2).after()"

    def __init__(self, targets: dict) -> None:
        """Requires list[func[(oldCall: str, oldArgs: str) -> (changed: bool, newPattern: str, newArgs: str)]]"""
        funcs = []
        for old, new in targets.items():

            def makeFunc(old, new):
                def func(oldCall: str, oldArgs: str):
                    if oldCall != old or "[" in oldArgs:
                        return False, None, None
                    return True, new, oldArgs

                return func

            funcs.append(makeFunc(old, new))
        self.parentPass = PassReplaceCallsGeneral(funcs)

    def visit(self, node: ast.AST) -> ast.AST:
        return self.parentPass.visit(node)


class PassReplaceCallsIfArgsWithoutNameOrList(NodeTransformerWithPrePost):
    # If pattern is an empty string, then replace the call with string constant "old() removed"
    # If pattern is "before().new({}).after()", then replace "old(1, 2)" with "before().new(1, 2).after()"

    def __init__(self, targets: dict) -> None:
        """Requires list[func[(oldCall: str, oldArgs: str) -> (changed: bool, newPattern: str, newArgs: str)]]"""
        funcs = []
        for old, new in targets.items():

            def makeFunc(old, new):
                def func(oldCall: str, oldArgs: str):
                    if oldCall != old:
                        return False, None, None
                    if oldCall != old or "[" in oldArgs:
                        return False, None, None
                    if oldArgs.upper() != oldArgs or oldArgs.lower() != oldArgs:
                        return False, None, None
                    return True, new, oldArgs

                return func

            funcs.append(makeFunc(old, new))
        self.parentPass = PassReplaceCallsGeneral(funcs)

    def visit(self, node: ast.AST) -> ast.AST:
        return self.parentPass.visit(node)


class PassReplaceCallsIfArgsEmpty(NodeTransformerWithPrePost):
    # If pattern is an empty string, then replace the call with string constant "old() removed"
    # If pattern is "before().new({}).after()", then replace "old(1, 2)" with "before().new(1, 2).after()"

    def __init__(self, targets: dict) -> None:
        """Requires list[func[(oldCall: str, oldArgs: str) -> (changed: bool, newPattern: str, newArgs: str)]]"""
        funcs = []
        for old, new in targets.items():

            def makeFunc(old, new):
                def func(oldCall: str, oldArgs: str):
                    if oldCall != old or oldArgs != "":
                        return False, None, None
                    return True, new, oldArgs

                return func

            funcs.append(makeFunc(old, new))
        self.parentPass = PassReplaceCallsGeneral(funcs)

    def visit(self, node: ast.AST) -> ast.AST:
        return self.parentPass.visit(node)


class PassReplaceCalls(NodeTransformerWithPrePost):
    # If pattern is an empty string, then replace the call with string constant "old() removed"
    # If pattern is "before().new({}).after()", then replace "old(1, 2)" with "before().new(1, 2).after()"

    def __init__(self, targets: dict) -> None:
        """Requires list[func[(oldCall: str, oldArgs: str) -> (changed: bool, newPattern: str, newArgs: str)]]"""
        funcs = []
        for old, new in targets.items():

            def makeFunc(old, new):
                def func(oldCall: str, oldArgs: str):
                    if oldCall != old:
                        return False, None, None
                    return True, new, oldArgs

                return func

            funcs.append(makeFunc(old, new))
        self.parentPass = PassReplaceCallsGeneral(funcs)

    def visit(self, node: ast.AST) -> ast.AST:
        return self.parentPass.visit(node)


class PassRemoveCalls(NodeTransformerWithPrePost):
    def __init__(self, targets: list) -> None:
        targetsDict = dict(
            zip(targets, ["'{}'".format(target + " removed") for target in targets])
        )
        self.parentPass = PassReplaceCalls(targetsDict)

    def visit(self, node: ast.AST) -> ast.AST:
        return self.parentPass.visit(node)


class PassRemoveImports(NodeTransformerWithPrePost):
    def visit_Import(self, node: ast.Import) -> ast.Expr:
        return ast.Expr(value=ast.Str(s=astunparse.unparse(node).strip("\n")))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.Expr:
        return ast.Expr(value=ast.Str(s=astunparse.unparse(node).strip("\n")))


class PassReplaceMeths(NodeTransformerWithPrePost):
    # If pattern is an empty string, then remove the method call entirely
    # If pattern is "before().new({}).after()", then replace ".old(1, 2)" with ".before().new(1, 2).after()"

    def __init__(self, targets: dict) -> None:
        self.targets = targets

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Attribute):
            for old, new in self.targets.items():
                if node.func.attr == old:
                    if new == "":
                        return node.func.value
                    prefix: str = astunparse.unparse(node.func.value).strip(
                        "\n"
                    )  # "a.b.c"
                    whole: str = astunparse.unparse(node).strip(
                        "\n"
                    )  # "a.b.c.old(1, 2)"
                    args: str = whole[len(prefix) + 1 + len(old) + 1 : -1]  # "1, 2"
                    newWhole: str = "{}.{}".format(
                        prefix, new
                    )  # "a.b.c.before().new({}).after()"
                    newWhole: str = newWhole.format(
                        args
                    )  # "a.b.c.before().new(1, 2).after()"
                    return ast.parse(newWhole).body[0].value

        self.generic_visit(node)
        return node


class PassRemoveMeths(NodeTransformerWithPrePost):
    def __init__(self, targets: list) -> None:
        self.targets = targets

    def visit(self, node: ast.AST) -> ast.AST:
        targetsDict = dict(zip(self.targets, [""] * len(self.targets)))
        passReplaceMeths = PassReplaceMeths(targetsDict)
        return passReplaceMeths.visit(node)


class PassRemoveTorchCuda(
    NodeTransformerWithPrePost
):  # remove .cuda() and device='cuda' in raw code
    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
            return node.func.value

        self.generic_visit(node)
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.keyword:
        if node.arg == "device":
            return None

        self.generic_visit(node)
        return node


class PassAppendTorchCuda(NodeTransformerWithPrePost):
    # For any assignment, appends .cuda() for:
    #   its lvalue, if its type (is torch.Tensor) or (includes "torch.nn.modules")

    @staticmethod
    def getCudaOneSinppet(varName: str) -> ast.AST:
        return ast.parse(
            """
if "torch.Tensor" in str(type({})) or "torch.nn.modules" in str(type({})):
    {} = {}.cuda()""".format(
                varName, varName, varName, varName
            )
        ).body[0]

    # def generic_visit(self, node: ast.AST) -> ast.AST:
    #     return generic_visit_with_pre_post(self, node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)
        for lval in node.targets:
            name: str = astunparse.unparse(lval)
            name = name.strip("\n")
            node.post.append(self.getCudaOneSinppet(name))
        return node


class PassCheckTorchInternalRandom(NodeTransformerWithPrePost):
    # Removes any known random number generator from the code

    def visit(self, node: ast.AST) -> ast.AST:
        passReplaceCalls = PassReplaceCalls(
            {
                "torch.randn": "torch.ones({})",
                "torch.rand": "torch.ones({})",
                "torch.rand_like": "torch.ones_like({})",
                "torch.randn_like": "torch.ones_like({})",
                "torch.empty": "torch.ones({})",
            }
        )
        node = passReplaceCalls.visit(node)

        passRemoveMeths = PassRemoveMeths(
            [
                "random_",
                "uniform_",
            ]
        )
        node = passRemoveMeths.visit(node)
        return node


class PassWithTfDevice(NodeTransformerWithPrePost):
    def __init__(self, device: str) -> None:
        self.device = device

    def wrapWith(self, oldBody: list) -> ast.With:
        node: ast.With = ast.parse(
            """with tf.device("{}"):
    pass""".format(
                self.device
            )
        ).body[0]
        node.body = oldBody
        return node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)

        nodeWith = self.wrapWith(node.body)
        node.body = [nodeWith]

        return node


class PassRandomizeTfInput(NodeTransformerWithPrePost):
    @staticmethod
    def getRandomSnippet(name: str):
        node: ast.With = ast.parse(
            """if "float" in str({0}.dtype):
    {0} = tf.random.normal({0}.shape)""".format(
                name
            )
        ).body[0]
        return node

    # def generic_visit(self, node: ast.AST) -> ast.AST:
    #     return generic_visit_with_pre_post(self, node)

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        self.generic_visit(node)

        if isinstance(node.value, ast.Call):
            callee: str = astunparse.unparse(node.value.func).strip("\n")
            if callee == "np.array" or callee == "tf.constant":
                if isinstance(node.targets[0], ast.Name):
                    randomSnippet = self.getRandomSnippet(node.targets[0].id)
                    node.post.append(randomSnippet)
        return node


class PassAddTfEagerCheck(NodeTransformerWithPrePost):
    @staticmethod
    def getEagerCheckSnippet():
        nodes: ast.With = ast.parse("if not tf.executing_eagerly():\n    exit(-1)").body
        return nodes

    def visit_Module(self, node: ast.Module) -> ast.Module:
        self.generic_visit(node)
        node.body.extend(self.getEagerCheckSnippet())
        return node


class SearchCall(NodeTransformerWithPrePost):
    """Search an specific API call."""

    def __init__(self, api_call: str, prefix: str):
        """
        for api `torch.nn.ConstantPad2d`, api_call = "ConstantPad2d", prefix = "nn"
        """
        self.api_call = api_call
        self.nodes = []
        self.prefix = prefix

    def search(self, o_ast) -> (int, list):
        self.nodes = []
        modified = self.visit(o_ast)
        return len(self.nodes), self.nodes

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
                return node

            self.nodes.append(node)
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
