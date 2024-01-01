import argparse
import astunparse
import json
import os
from pathlib import Path
from typing import List
import ast
import traceback
import torch
from torch_utils import test_wrapper

from constant.returntypes import ResType


OUT_DIR, RESULT_DIR, TEST_DIR, TEST_LOG_PATH, TEMP_LOG_PATH, DEVICE = (
    None,
    None,
    None,
    None,
    None,
    None,
)
COV = False
CODE_PARSER = None
TEST_WRAPPER = None

OUTPUT_LIMIT: int = 1024
SEED: int = 420
MATCH_COV_FILE = Path("/tmp/match_trigger.log")
MAXIMUM_TESTCASES = 10


def clean_match_cov():
    MATCH_COV_FILE.write_text("")


def get_match_cov():
    cov = []
    if MATCH_COV_FILE.exists():
        cov = MATCH_COV_FILE.read_text().splitlines()
    clean_match_cov()
    return cov


class MultilineAssignTransformer(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Tuple):
            if len(node.targets[0].elts) == len(node.value.elts):
                return [
                    ast.Assign(targets=[t], value=v)
                    for t, v in zip(node.targets[0].elts, node.value.elts)
                ]
        return node


class LibAssignRemover(ast.NodeTransformer):
    def __init__(self, lib_name: str = "torch") -> None:
        super().__init__()
        self.lib_name = lib_name

    def visit_Assign(self, node):
        if any(self.is_lib_attribute(target) for target in node.targets):
            return ast.Pass()
        return self.generic_visit(node)

    def is_lib_attribute(self, node):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == self.lib_name:
                return True
            return self.is_lib_attribute(node.value)
        return False


class CodeParser:
    def __init__(self, lib_name: str = "torch") -> None:
        self.transformers = [MultilineAssignTransformer(), LibAssignRemover(lib_name)]
        self.lib_name = lib_name
        if lib_name == "torch":
            self.is_input = lambda x: torch.is_tensor(x)
            self.imports = (
                "import os\nimport torch\nimport torch.nn.functional as F\nimport torch.nn as nn\n"
                "import numpy as np\nfrom torch.autograd import Variable\nimport math\n"
                "import torch as th\nimport torch.linalg as la\n"
                "from torch.nn import Parameter\n"
                "import torch.linalg as linalg\n"
            )
            self._init_code = "{} = torch.randn(1, 1, 1)\n"
        elif lib_name == "tf":
            raise NotImplementedError
        else:
            raise NotImplementedError

    def input_init_code(self, arg_name):
        return self._init_code.format(arg_name)

    def split_func_tensor(self, code):
        # get the code of model
        code = self.preprocessing(code)
        tree = ast.parse(code)

        class_init_args = []
        class_init_required_args = []
        class_init_code = ""

        class_code = ""
        class_name = ""

        class_forward_args = []
        class_forward_required_args = []

        inputs: List[str] = []
        input_init_code = ""

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_code += ast.unparse(node) + "\n\n"
                class_name = node.name

                # get the arguments the initiation of this class
                try:
                    init_method = next(
                        node
                        for node in ast.walk(node)
                        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
                    )

                    class_init_args = [arg.arg for arg in init_method.args.args[1:]]
                    defaults = init_method.args.defaults
                    class_init_required_args = class_init_args[
                        : len(class_init_args) - len(defaults)
                    ]
                except Exception as e:
                    pass

                try:
                    forward_method = next(
                        node
                        for node in ast.walk(node)
                        if isinstance(node, ast.FunctionDef) and node.name == "forward"
                    )
                    class_forward_args = [
                        arg.arg for arg in forward_method.args.args[1:]
                    ]
                    defaults = forward_method.args.defaults
                    class_forward_required_args = class_forward_args[
                        : len(class_forward_args) - len(defaults)
                    ]
                except Exception as e:
                    pass

            elif isinstance(node, ast.Assign):
                value = node.value
                if isinstance(value, ast.Call):
                    # first check whether is initialization of the class
                    if isinstance(value.func, ast.Name) and value.func.id == class_name:
                        # first split the tensor arguments and non-tensor arguments
                        if len(value.args) >= len(class_init_required_args) and len(
                            value.args
                        ) <= len(class_init_args):
                            class_init_code = (
                                "func = " + ast.unparse(value) + f".to('{DEVICE}')\n"
                            )
                        else:
                            class_init_code = ""
                        continue

                    func = value.func
                    args = value.args

                    try:
                        tgt = node.targets[0].id
                    except Exception as e:
                        continue

                    init_code = ast.unparse(node)
                    if tgt not in inputs:
                        # we need the arg code
                        for arg in ast.walk(value):
                            if isinstance(arg, ast.Name):
                                init_code = (
                                    self.find_name_in_tree(tree, arg.id)
                                    + "\n"
                                    + init_code
                                )
                            elif isinstance(arg, ast.Starred):
                                if isinstance(arg.value, ast.Name):
                                    init_code = (
                                        self.find_name_in_tree(tree, arg.value.id)
                                        + "\n"
                                        + init_code
                                    )

                        # test whether is tensor
                        try:
                            exec(init_code)
                            if self.is_input(eval(tgt)):
                                inputs.append(tgt)
                                input_init_code += init_code + "\n"
                            elif tgt in class_forward_args:
                                inputs.append(tgt)
                                input_init_code += init_code + "\n"
                        except Exception as e:
                            pass

        class_init_args_code = ""
        for arg_name in class_init_required_args:
            class_init_args_code += (
                self.find_name_in_tree(tree, arg_name, use_default=True) + "\n"
            )
        if class_init_code != "":
            class_init_code = class_init_args_code + class_init_code
        else:
            class_init_code = class_init_args_code
            class_init_code += f"\nfunc = {class_name}({', '.join(class_init_required_args)}).to('{DEVICE}')\n"
        class_code += "\n" + class_init_code

        if len(inputs) < len(class_forward_args):
            diff = len(class_forward_args) - len(inputs)
            for arg_name in class_forward_required_args:
                if arg_name not in inputs:
                    inputs.append(arg_name)
                    input_init_code += f"{arg_name} = 1\n"
                    diff -= 1
                    if diff == 0:
                        break

        return class_code, inputs, input_init_code

    def preprocessing(self, code: str):
        code = code.replace("\t", "    ")

        new_lines = []
        for line in code.splitlines():
            if line.strip().startswith("assert"):
                continue
            new_lines.append(line)
        code = "\n".join(new_lines)

        tree = ast.parse(code)
        for transformer in self.transformers:
            tree = transformer.visit(tree)
        code = astunparse.unparse(tree)
        code = code.replace("(:", ":").replace(":)", ":")
        return code

    @staticmethod
    def find_name_in_tree(tree, arg_name, use_default=False):
        for _n in tree.body:
            if isinstance(_n, ast.Assign):
                for _t in _n.targets:
                    if isinstance(_t, ast.Name) and _t.id == arg_name:
                        return ast.unparse(_n)
        if arg_name == "batch_size":
            return f"{arg_name} = 1"

        if use_default:
            return f"{arg_name} = 1"
        else:
            return ""


def _cross_check(func_def_code, tensors, filename):
    func_def_code += f"test_inputs = [{', '.join(tensors)}]\n"
    TEMP_LOG_PATH.write_text(func_def_code)

    if COV:
        clean_match_cov()

    result, errors = eval(f"TEST_WRAPPER(func_def_code, 420, tensors, '{DEVICE}')")

    if COV:
        match_info = {filename: get_match_cov()}
        with open(TEST_DIR / "match.log", "a") as fw:
            fw.write(json.dumps(match_info) + "\n")

    error_msg = "\n".join([f"{k}:\n{v}\n" for k, v in errors.items()])
    error_msg = "\n'''\n" + error_msg + "'''"

    # print(msg)
    if result == ResType.PASS:
        with open(TEST_DIR / "success.log", "a") as fw:
            fw.write(filename + "\n")
        raise Exception("Success", "succeed")
    elif result == ResType.NAN:
        with open(TEST_DIR / "nan.log", "a") as fw:
            fw.write(filename + "\n")
        raise Exception("NAN", "void")
    elif result == ResType.RANDOM:
        with open(TEST_DIR / "random.log", "a") as fw:
            fw.write(filename + "\n")
        raise Exception("RANDOM", "void")
    elif result == ResType.SKIP:
        skip_dir = RESULT_DIR / "skip"
        os.makedirs(skip_dir, exist_ok=True)
        with open(f"{skip_dir}/{filename}", "w") as fw:
            fw.write(func_def_code)
        with open(TEST_DIR / "skip.log", "a") as fw:
            fw.write(filename + "\n")

        raise Exception("SKIP", "void")
    else:
        exception_name = str(result).replace("ResType.", "")
        bug_dir = RESULT_DIR / exception_name.lower()

        os.makedirs(bug_dir, exist_ok=True)
        with open(TEST_DIR / f"{exception_name.lower()}.log", "a") as fw:
            fw.write(filename + "\n")

        with open(f"{bug_dir}/{filename}", "w") as fw:
            fw.write(func_def_code + "\n# " + exception_name + error_msg)
        raise Exception(exception_name, "Catch")


def validate(func_def_code, tensors, filename):
    func_def_code += f"test_inputs = [{', '.join(tensors)}]\n"
    TEMP_LOG_PATH.write_text(func_def_code)

    result, errors = eval(
        f"TEST_WRAPPER(func_def_code, 420, tensors, '{DEVICE}', 'validate')"
    )
    if result == ResType.PASS:
        with open(TEST_DIR / "success.log", "a") as fw:
            fw.write(filename + "\n")
    else:
        with open(TEST_DIR / "fail.log", "a") as fw:
            fw.write(filename + "\n")


def read_all_tasks():
    tasks = []
    for opt_dir in OUT_DIR.iterdir():
        if not opt_dir.is_dir():
            continue
        opt = opt_dir.name
        for filename in opt_dir.iterdir():
            if not filename.name.endswith(".py"):
                continue

            label = filename.name[:-3]
            code = filename.read_text()
            tasks.append([opt, label, code])

    tasks = sorted(tasks, key=lambda x: (x[0], x[1]))
    return tasks


def core_oracle(code, filename, is_validate=False):
    class_def_code, inputs, input_init_code = CODE_PARSER.split_func_tensor(code)
    imports = CODE_PARSER.imports

    if len(inputs) == 0:
        inputs.append("input_tensor")
        input_init_code += CODE_PARSER.input_init_code("input_tensor")

    class_def_code = imports + "\n" + class_def_code + "\n" + input_init_code + "\n"

    if is_validate:
        validate(class_def_code, inputs, filename)
    else:
        _cross_check(class_def_code, inputs, filename)


def core_loop(args):
    tasks = read_all_tasks()
    try:
        tested = set(open(TEST_LOG_PATH, "r").read().splitlines())
    except Exception:
        tested = set([])

    count = 0
    for id in range(len(tasks)):
        task = tasks[id]
        api, label, code = task
        filename = label + ".py"

        if filename in tested:
            continue
        with open(TEST_LOG_PATH, "a") as fw:
            fw.write(filename + "\n")

        try:
            core_oracle(code, filename, is_validate=args.validate)
        except Exception as e:
            reason: str = "FrameworkCrashCatch"
            detail: str = str(e)
            if len(e.args) >= 2:
                reason: str = e.args[0]
                detail: str = e.args[1]

            if (
                reason == "FrameworkCrashCatch"
            ):  # FrameworkCrashCatch is printed by driver
                print(traceback.format_exc())
                print(detail)
                exit(-1)

            if "Catch" in reason:
                with open("catches.log", "a") as f:
                    f.write(
                        "\nLmfuzzTestcase {} {} {} {} {} {}".format(
                            id, api, label, reason, SEED, detail
                        )
                    )
            print("\nLmfuzzTestcase", id, api, label, reason, SEED, detail)
            print("----------------------------------\n")

        count += 1
        if count >= MAXIMUM_TESTCASES:
            exit(123)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--lib", type=str, default="torch")

    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--res_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--test_log_path", type=str, default=None)
    parser.add_argument("--temp_log_path", type=str, default=None)
    parser.add_argument("--cov", action="store_true", default=False)
    parser.add_argument("--validate", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    lib = args.lib

    global OUT_DIR, RESULT_DIR, TEST_DIR, TEST_LOG_PATH, TEMP_LOG_PATH, DEVICE, COV
    OUT_DIR = Path(args.out_dir)
    RESULT_DIR = Path(args.res_dir)
    TEST_DIR = Path(args.test_dir)
    TEST_LOG_PATH = Path(args.test_log_path)
    TEMP_LOG_PATH = Path(args.temp_log_path)
    DEVICE = args.device
    COV = args.cov

    RESULT_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)

    global CODE_PARSER, TEST_WRAPPER
    if lib == "torch":
        CODE_PARSER = CodeParser(lib_name="torch")
        TEST_WRAPPER = test_wrapper
    else:
        # TODO: add support for other libraries
        raise NotImplementedError(f"Library {lib} is not supported yet")

    if COV and lib == "torch":
        global MATCH_COV_FILE
        MATCH_COV_FILE = Path("/", "tmp", f"trigger-{RESULT_DIR.name}")
        torch.version.log_path = str(MATCH_COV_FILE)

    core_loop(args)


if __name__ == "__main__":
    main()
    # Some sneaky code may contain exit(0) or other equivalent calls
    # We distinguish ourselves from them with a magic number
    exit(233)
