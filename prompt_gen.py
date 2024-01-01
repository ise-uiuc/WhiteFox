import json
import argparse
from pathlib import Path
from typing import List
import numpy as np
from prompt_template import _trigger_template_no_code, _pattern_template_no_code
import random
import os

feedback_template = """# Model begins
{}
# Inputs to the model
{}
# Model ends
"""


def select_examples(examples, num, use_rl):
    length = len(examples)
    if num > length:
        idxs = list(range(length))
    elif not use_rl:
        idxs = random.sample(range(length), num)
    else:
        beta_list = []
        for e in examples:
            alpha = e["alpha"] if "alpha" in e else 1
            beta = e["beta"] if "beta" in e else 1
            beta_list.append(np.random.beta(alpha, beta))
        idxs = np.argsort(beta_list)[-num:].tolist()
    return idxs


class Optim:
    def __init__(self):
        pass

    def get_opts(self):
        return self.opts.keys()

    def _target_line(self, opt):
        return self.opts[opt]["target_line"]

    def _opt_name(self, opt):
        return opt

    def _src_code(self, code_paths, use_mini=False):
        code = ""
        for code_path in code_paths:
            if use_mini:
                code_path = code_path.replace("-full", "-mini")
            code += open(code_path, "r").read() + "\n"
        return code

    def _join(self, fillings):
        if len(fillings) == 1:
            return fillings[0]

        idx = 1
        res = "\n"
        for filling in fillings[:-1]:
            res += "{}) {}, ".format(idx, filling)
            idx += 1
        res += "and {}) {}".format(idx, fillings[-1])
        return res

    def _join_info(self, given_info):
        if len(given_info) == 1:
            return given_info[0]

        res = ""
        for info in given_info[:-1]:
            res += info + ", "
        res += "and " + given_info[-1]
        return res

    def _get_hint_code(self, hint, use_mini=False):
        code = ""
        fillings = []
        if hint["type"] == "trigger":
            fillings += [
                _trigger_template_no_code.format(hint["target_line"], hint["func"])
            ]
            code += (
                f"# Code of the function `{hint['func']}` and its helper functions\n"
            )
            code += self._src_code(hint["codes"], use_mini)
            code += "\n"
        elif hint["type"] == "nl":
            raise NotImplementedError
        elif hint["type"] == "pattern":
            fillings += [_pattern_template_no_code]
            code += f"# Code of the pattern:\n"
            code += self._src_code(hint["codes"])
            code += "\n"
        else:
            raise NotImplementedError
        return code, fillings

    def _indent(self, code):
        return "\n".join(["    " + line for line in code.splitlines()])

    def _load_nl(self, nl_path, idx):
        nls = {}
        for opt_dir in Path(nl_path).iterdir():
            if not opt_dir.is_dir():
                continue
            try:
                nl = (opt_dir / f"{opt_dir.name}_{idx}.txt").read_text().strip()
                # check whether the opt is in the opts
                nls[opt_dir.name] = nl
            except:
                pass
        return nls

    def get_prompt(self, template, opt, use_mini=False):
        raise NotImplementedError


class NL(Optim):
    def __init__(self, opt_path, file_path, idx=1):
        super().__init__()
        opts = json.loads(open(opt_path, "r").read())
        nl = self._load_nl(file_path, idx)
        self.opts = {}
        for k in opts:
            if k in nl:
                self.opts[k] = nl[k]
            else:
                raise ValueError(f"Cannot find NL for {k} in {file_path}")

    def get_prompt(self, template, opt, use_mini=False):
        return template.format(self.opts[opt])


class Src2Test(Optim):
    def __init__(self, file_path):
        super().__init__()
        self.opts = json.loads(open(file_path, "r").read())

    def get_prompt(self, template, opt, use_mini=False):
        opt_info = self.opts[opt]
        hints = opt_info["hints"]

        fillings = []
        code = ""
        for hint in hints:
            _code, _fillings = self._get_hint_code(hint, use_mini)
            code += _code
            fillings += _fillings

        fillings = self._join(fillings)
        return template.format(fillings, code.strip())


class Src2TestTFLite(Src2Test):
    def _get_hint_code(self, hint, use_mini=False):
        return self._src_code(hint["codes"], use_mini)

    def get_prompt(self, template, opt, use_mini=False):
        opt_info = self.opts[opt]
        hints = opt_info["hints"]

        code = ""
        for hint in hints:
            _code = self._get_hint_code(hint, use_mini)
            code += _code

        return template.format(code.strip())


class Src2NLTFLite(Src2TestTFLite):
    PLACEHOLDER_TFLITE_OPTIMIZATION_NAME = "PLACEHOLDER_TFLITE_OPTIMIZATION_NAME"
    PLACEHOLDER_SRC_CODE = "PLACEHOLDER_SRC_CODE"

    def get_prompt(self, template, opt, use_mini=False):
        opt_info = self.opts[opt]
        hints = opt_info["hints"]

        code = ""
        for hint in hints:
            _code = self._get_hint_code(hint, use_mini)
            code += _code
        code_formatted = f"```cpp\n{code.strip()}\n```"
        # Cannot use template.format because c++ code contains { ... }
        prompt = template.replace(self.PLACEHOLDER_TFLITE_OPTIMIZATION_NAME, opt)
        prompt = prompt.replace(self.PLACEHOLDER_SRC_CODE, code_formatted)
        return prompt


class Src2NLTFXLA(Src2NLTFLite):
    PLACEHOLDER_TFLITE_OPTIMIZATION_NAME = "PLACEHOLDER_TFXLA_OPTIMIZATION_NAME"
    PLACEHOLDER_SRC_CODE = "PLACEHOLDER_SRC_CODE"
    PLACEHOLDER_TARGET_LINE = "PLACEHOLDER_TARGET_LINE"
    PLACEHOLDER_FUNC_NAME = "PLACEHOLDER_FUNC_NAME"

    def format_source_code(self, code: str) -> str:
        """Format the source code in a code block."""
        return f"```cpp\n{code.strip()}\n```"

    def get_prompt(self, template, opt, use_mini=False):
        opt_info = self.opts[opt]
        hints = opt_info["hints"]

        code = ""
        func_name = ""
        target_line = ""
        for hint in hints:
            _code = self._get_hint_code(hint, use_mini)
            code += _code
            func_name = hint["func"]
            target_line = hint["target_line"]
        if target_line not in code:
            print(f"[WARNING] {opt} target line {target_line} does not exist.")
        # Cannot use template.format because c++ code contains { ... }
        prompt = template.replace(self.PLACEHOLDER_TFLITE_OPTIMIZATION_NAME, opt)
        prompt = prompt.replace(
            self.PLACEHOLDER_SRC_CODE, self.format_source_code(code)
        )
        prompt = prompt.replace(self.PLACEHOLDER_TARGET_LINE, target_line)
        prompt = prompt.replace(self.PLACEHOLDER_FUNC_NAME, func_name)

        return prompt


class SrcNLTest2Template(Optim):
    def __init__(self, file_path, nl_path, test_path, idx=1, use_rl=False):
        super().__init__()
        self.opts = json.loads(open(file_path, "r").read())
        self.nls = self._load_nl(nl_path, idx)
        self.tests = self._load_test(test_path)
        self.use_rl = use_rl
        # assert(len(self.opts) <= len(self.nls))
        for opt in self.opts:
            assert opt in self.nls

    def _load_test(self, test_path):
        tests = json.loads(open(test_path, "r").read())
        return tests

    def get_prompt(self, template, opt, use_mini=False, num_of_prompts=10, num_model=2):
        if opt not in self.tests:
            # only for the triggered ones
            return []

        opt_info = self.opts[opt]
        hints = opt_info["hints"]

        fillings = []
        code = ""
        for hint in hints:
            _code, _fillings = self._get_hint_code(hint, use_mini)
            code += _code
            fillings += _fillings

        fillings = self._join(fillings)
        nl = self.nls[opt]

        output_prompts = []
        for _ in range(num_of_prompts):
            idxs = select_examples(self.tests[opt], num_model, use_rl=self.use_rl)
            self.tests[f"{opt}_selected"] = idxs
            infill_tests = []
            for i in idxs:
                model_code_lines = []
                model_code = self.tests[opt][i]["model_code"]
                for line in model_code.splitlines():
                    if line.startswith("func ="):
                        continue
                    if line.strip() == "":
                        continue
                    model_code_lines.append(line)
                model_code = "\n".join(model_code_lines).strip()

                input_code_lines = []
                input_code = self.tests[opt][i]["input_code"].strip()
                for line in input_code.splitlines():
                    if line == "":
                        continue
                    input_code_lines.append(line)
                input_code = "\n".join(input_code_lines).strip()

                infill_tests.append(feedback_template.format(model_code, input_code))
            infill_code = "\n".join(infill_tests)
            output_prompts.append(
                template.format(fillings, code.strip(), nl.strip(), infill_code)
            )
        return output_prompts


class NLTest2Template(Optim):
    def __init__(self, file_path, nl_path, test_path, idx=1, use_rl=False):
        super().__init__()
        self.opts = json.loads(open(file_path, "r").read())
        self.nls = self._load_nl(nl_path, idx)
        self.tests = self._load_test(test_path)
        self.use_rl = use_rl
        assert len(self.opts) <= len(self.nls)

    def _load_test(self, test_path):
        # TODO: load test
        tests = json.loads(open(test_path, "r").read())
        return tests

    def get_prompt(self, template, opt, use_mini=False, num_of_prompts=10, num_model=2):
        if opt not in self.tests:
            # only for the triggered ones
            return []

        nl = self.nls[opt]

        output_prompts = []
        for _ in range(num_of_prompts):
            idxs = select_examples(self.tests[opt], num_model, use_rl=self.use_rl)
            self.tests[f"{opt}_selected"] = idxs
            infill_tests = []
            for i in idxs:
                model_code_lines = []
                model_code = self.tests[opt][i]["model_code"]
                for line in model_code.splitlines():
                    if line.startswith("func ="):
                        continue
                    if line.strip() == "":
                        continue
                    model_code_lines.append(line)
                model_code = "\n".join(model_code_lines).strip()

                input_code_lines = []
                input_code = self.tests[opt][i]["input_code"].strip()
                for line in input_code.splitlines():
                    if line == "":
                        continue
                    input_code_lines.append(line)
                input_code = "\n".join(input_code_lines).strip()

                infill_tests.append(feedback_template.format(model_code, input_code))
            infill_code = "\n".join(infill_tests)
            output_prompts.append(template.format(nl.strip(), infill_code))
        return output_prompts


class SrcNL2Test(Optim):
    def __init__(self, file_path, nl_path, idx=1):
        super().__init__()
        self.opts = json.loads(open(file_path, "r").read())
        self.nls = self._load_nl(nl_path, idx)
        assert len(self.opts) <= len(self.nls)

    def get_prompt(self, template, opt, use_mini=False):
        opt_info = self.opts[opt]
        hints = opt_info["hints"]

        fillings = []
        code = ""
        for hint in hints:
            _code, _fillings = self._get_hint_code(hint, use_mini)
            code += _code
            fillings += _fillings

        fillings = self._join(fillings)
        nl = self.nls[opt]
        return template.format(fillings, code.strip(), nl.strip())


class SrcNL2TestTFLite(SrcNL2Test):
    PLACEHOLDER_OPTIM_NAME = "PLACEHOLDER_TFLITE_OPTIMIZATION_NAME"
    PLACEHOLDER_SRC_CODE = "PLACEHOLDER_SRC_CODE"
    PLACEHOLDER_DESC = "PLACEHOLDER_DESC"

    def __init__(self, file_path, nl_path, idx=1):
        self.opts = json.loads(open(file_path, "r").read())
        self.nls = self._load_nl(nl_path, idx)

    def _get_code(self, hint, use_mini=False):
        return self._src_code(hint["codes"], use_mini)

    def get_optim_source_code(self, opt: str, use_mini=False) -> str:
        opt_info = self.opts[opt]
        hints = opt_info["hints"]

        code = ""
        for hint in hints:
            _code = self._get_code(hint, use_mini)
            code += _code
        return code

    def get_prompt(self, template, opt, use_mini=False):
        if opt not in self.nls:
            return None
        nl = self.nls[opt]
        code = self.get_optim_source_code(opt, use_mini=use_mini)

        prompt = template.replace(self.PLACEHOLDER_OPTIM_NAME, opt)
        prompt = prompt.replace(self.PLACEHOLDER_SRC_CODE, code)
        prompt = prompt.replace(self.PLACEHOLDER_DESC, nl)
        return prompt


class SrcNL2TestTFXLA(SrcNL2TestTFLite):
    PLACEHOLDER_OPTIM_NAME = "PLACEHOLDER_TFXLA_OPTIMIZATION_NAME"
    PLACEHOLDER_SRC_CODE = "PLACEHOLDER_SRC_CODE"
    PLACEHOLDER_DESC = "PLACEHOLDER_DESC"
    PLACEHOLDER_TARGET_LINE = "PLACEHOLDER_TARGET_LINE"
    PLACEHOLDER_FUNC_NAME = "PLACEHOLDER_FUNC_NAME"

    def format_source_code(self, code: str) -> str:
        """Format the source code in a code block."""
        return f"```cpp\n{code.strip()}\n```"

    def get_optim_source_code(self, opt: str, use_mini=False) -> str:
        opt_info = self.opts[opt]
        hints = opt_info["hints"]

        code = ""
        for hint in hints:
            _code = self._get_code(hint, use_mini)
            code += _code
        code = self.format_source_code(code)
        return code

    def get_prompt(self, template, opt, use_mini=False):
        if opt not in self.nls:
            return None
        nl = self.nls[opt]
        code = self.get_optim_source_code(opt, use_mini=use_mini)

        hints = self.opts[opt]["hints"]
        func_name = ""
        target_line = ""
        for hint in hints:
            func_name = hint["func"]
            target_line = hint["target_line"]

        prompt = template.replace(self.PLACEHOLDER_OPTIM_NAME, opt)
        prompt = prompt.replace(self.PLACEHOLDER_SRC_CODE, code)
        prompt = prompt.replace(self.PLACEHOLDER_DESC, nl)
        prompt = prompt.replace(self.PLACEHOLDER_TARGET_LINE, target_line)
        prompt = prompt.replace(self.PLACEHOLDER_FUNC_NAME, func_name)

        return prompt


class SrcNL2TestFeedbackTFLite(SrcNL2TestTFLite):
    PLACEHOLDER_OPTIM_NAME = "PLACEHOLDER_TFLITE_OPTIMIZATION_NAME"
    PLACEHOLDER_SRC_CODE = "PLACEHOLDER_SRC_CODE"
    PLACEHOLDER_DESC = "PLACEHOLDER_DESC"
    PLACEHOLDER_MODEL_EXAMPLE = "PLACEHOLDER_MODEL_EXAMPLES"
    EXAMPLE_TEMPLATE = "# Model begins\nPLACEHOLDER_MODEL\n# Model ends"

    def __init__(self, file_path, nl_path, idx=1, examples=None, k_shot=3):
        self.opts = json.loads(open(file_path, "r").read())
        self.nls = self._load_nl(nl_path, idx)
        self.examples = examples
        self.k_shot = k_shot

    def _get_code(self, hint, use_mini=False):
        return self._src_code(hint["codes"], use_mini)

    def get_examples(self, opt):
        # If no examples? better use the 1shot one
        # if self.exampels[opt] == []:
        #     return None
        # If not enough examples, simple
        opt_examples = self.examples[opt]
        if len(self.examples[opt]) <= self.k_shot:
            return "\n\n".join(
                [
                    self.EXAMPLE_TEMPLATE.replace("PLACEHOLDER_MODEL", code)
                    for code in opt_examples
                ]
            )

        ids = np.random.choice(len(opt_examples), self.k_shot, replace=False)
        return "\n\n".join(
            [
                self.EXAMPLE_TEMPLATE.replace("PLACEHOLDER_MODEL", opt_examples[id])
                for id in ids
            ]
        )

    def get_prompt(self, template, opt, use_mini=False):
        if opt not in self.nls:
            return None

        opt_info = self.opts[opt]
        hints = opt_info["hints"]

        code = ""
        for hint in hints:
            _code = self._get_code(hint, use_mini)
            code += _code

        nl = self.nls[opt]
        examples = self.get_examples(opt)
        if examples is None:
            return None
        prompt = template.replace(self.PLACEHOLDER_OPTIM_NAME, opt)
        prompt = prompt.replace(self.PLACEHOLDER_SRC_CODE, code)
        prompt = prompt.replace(self.PLACEHOLDER_DESC, nl)
        prompt = prompt.replace(self.PLACEHOLDER_MODEL_EXAMPLE, examples)
        return prompt


class SrcNLTest2TemplateTFLite(SrcNLTest2Template):
    PLACEHOLDER_OPTIM_NAME = "PLACEHOLDER_TFLITE_OPTIMIZATION_NAME"
    PLACEHOLDER_SRC_CODE = "PLACEHOLDER_SRC_CODE"
    PLACEHOLDER_DESC = "PLACEHOLDER_DESC"
    PLACEHOLDER_MODEL_EXAMPLE = "PLACEHOLDER_MODEL_EXAMPLES"
    EXAMPLE_TEMPLATE = "# Model begins\nPLACEHOLDER_EXAMPLE_MODEL\n# Model ends"

    def _load_test(self, test_path):
        if not os.path.exists(test_path):
            print("[WARNING] trigger test path not exists.")
            return dict()
        try:
            # TODO: load test
            tests = json.loads(open(test_path, "r").read())
        except Exception as e:
            print("[ERROR]", str(e))
            tests = {}
        return tests

    def _get_code(self, hint, use_mini=False):
        return self._src_code(hint["codes"], use_mini)

    def _format_model_code(self, model_code: str) -> str:
        model_code_lines = []
        for line in model_code.splitlines():
            if line.startswith("func ="):
                continue
            if line.strip() == "":
                continue
            model_code_lines.append(line)
        return "\n".join(model_code_lines).strip()

    def _format_input_code(self, input_code: str) -> str:
        input_code_lines = []
        for line in input_code.splitlines():
            if line == "":
                continue
            input_code_lines.append(line)
        return "\n".join(input_code_lines).strip()

    def get_triggering_tests(self, opt: str):
        if "trigger_tests" not in self.tests:
            return []
        if opt not in self.tests["trigger_tests"]:
            # If there's no triggered test in prior steps, return an empty list
            return []
        return self.tests["trigger_tests"][opt]

    def get_prompt(
        self, template: str, opt: str, use_mini=False, num_of_prompts=1, num_model=3
    ) -> List[str]:
        opt_triggering_tests = self.get_triggering_tests(opt)
        if len(opt_triggering_tests) == 0:
            return []

        opt_info = self.opts[opt]
        hints = opt_info["hints"]

        optim_src_code = ""
        for hint in hints:
            _code = self._get_code(hint, use_mini)
            optim_src_code += _code
        nl = self.nls[opt]

        output_prompts = []
        for _ in range(num_of_prompts):
            idxs = select_examples(opt_triggering_tests, num_model, use_rl=self.use_rl)
            self.tests[f"{opt}_selected"] = idxs
            example_models = []
            for i in idxs:
                # model_code = self._format_model_code(self.tests[opt][i]['model_code'])

                # input_code = self._format_input_code(self.tests[opt][i]['input_code'].strip())

                # example_models.append(feedback_template.format(model_code, input_code))
                code = opt_triggering_tests[i]["code"].strip()
                example_models.append(
                    self.EXAMPLE_TEMPLATE.replace("PLACEHOLDER_EXAMPLE_MODEL", code)
                )
            examples: str = "\n\n".join(example_models)
            prompt = template.replace(self.PLACEHOLDER_OPTIM_NAME, opt)
            prompt = prompt.replace(self.PLACEHOLDER_SRC_CODE, optim_src_code)
            prompt = prompt.replace(self.PLACEHOLDER_DESC, nl)
            prompt = prompt.replace(self.PLACEHOLDER_MODEL_EXAMPLE, examples)
            output_prompts.append(prompt)
        return output_prompts


class SrcNLTest2TemplateTFXLA(SrcNLTest2TemplateTFLite):
    PLACEHOLDER_OPTIM_NAME = "PLACEHOLDER_TFXLA_OPTIMIZATION_NAME"

    def get_triggering_tests(self, opt: str):
        if "trigger_tests" not in self.tests:
            return []
        if opt not in self.tests["trigger_tests"]:
            try:
                opt_info = self.opts[opt]
                hints = opt_info["hints"]
                opt_alias = hints[0]["codes"][0].rsplit("/", 1)[-1][:-3]
                print(f"[{opt}] ~ [{opt_alias}]")
                if opt_alias in self.tests["trigger_tests"]:
                    return self.tests["trigger_tests"][opt_alias]
            except:
                return []
            # If there's no triggered test in prior steps, return an empty list
            return []
        return self.tests["trigger_tests"][opt]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optpath", type=str, default="")
    parser.add_argument("--nlpath", type=str, default="")
    parser.add_argument("--testpath", type=str, default="")
    parser.add_argument("--mode", type=str, default="src2test")
    parser.add_argument("--nlidx", type=int, default=1)
    parser.add_argument("--template", type=str, default="template/demo.txt")
    parser.add_argument("--outdir", type=str, default="prompt/demo")
    parser.add_argument(
        "--lib", type=str, default="torch", choices=["torch", "tf", "tfxla"]
    )
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--step1dir", type=str, default="prompt/demo/step1")
    parser.add_argument("--num_model", type=int, default=2)
    parser.add_argument("--use_rl", action="store_true", default=False)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    template = open(args.template, "r").read()
    use_rl = args.use_rl

    if args.mode == "src2test":
        if args.lib == "tf":
            optim = Src2TestTFLite(args.optpath)
        else:
            optim = Src2Test(args.optpath)
        for opt in optim.get_opts():
            prompt = optim.get_prompt(template, opt)
            with open(outdir / f"{opt}.txt", "w") as f:
                f.write(prompt)
    elif args.mode == "src2nl":
        if args.lib == "tf":
            # Use mini by default.
            assert args.mini
            optim = Src2NLTFLite(args.optpath)
        elif args.lib == "tfxla":
            assert args.mini
            optim = Src2NLTFXLA(args.optpath)
        else:
            optim = Src2Test(args.optpath)
        for opt in optim.get_opts():
            prompt = optim.get_prompt(template, opt, args.mini)
            with open(outdir / f"{opt}.txt", "w") as f:
                f.write(prompt)
    elif args.mode == "srcnl2test":
        if args.lib == "tf":
            optim = SrcNL2TestTFLite(args.optpath, args.nlpath, args.nlidx)
        elif args.lib == "tfxla":
            optim = SrcNL2TestTFXLA(args.optpath, args.nlpath, args.nlidx)
        else:
            optim = SrcNL2Test(args.optpath, args.nlpath, args.nlidx)

        for opt in optim.get_opts():
            prompt = optim.get_prompt(template, opt)
            if prompt is None:
                continue
            with open(outdir / f"{opt}.txt", "w") as f:
                f.write(prompt)
    elif args.mode == "srcnl2test_feedback":
        if args.lib == "tf":
            # Use mini by default.
            assert args.mini

            def _load_trigger():
                import os

                data_root_dir = "output/starcoder/tflite/tflite-0601-trigger"
                examples = dict()
                for run_id in os.listdir(data_root_dir):
                    for optim_name in os.listdir(os.path.join(data_root_dir, run_id)):
                        if optim_name not in examples:
                            examples[optim_name] = []
                        for fname in os.listdir(
                            os.path.join(data_root_dir, run_id, optim_name)
                        ):
                            code = (
                                open(
                                    os.path.join(
                                        data_root_dir, run_id, optim_name, fname
                                    ),
                                    "r",
                                )
                                .read()
                                .strip()
                            )
                            examples[optim_name].append(code)
                for optim_name in examples:
                    examples[optim_name] = list(set(examples[optim_name]))
                return examples

            trigger_examples = _load_trigger()
            optim = SrcNL2TestFeedbackTFLite(
                args.optpath, args.nlpath, args.nlidx, examples=trigger_examples
            )
        for opt in optim.get_opts():
            # Generate 100 prompts
            for i in range(100):
                prompt = optim.get_prompt(template, opt)
                if prompt is None:
                    continue
                with open(outdir / f"{opt}_{i}.txt", "w") as f:
                    f.write(prompt)
    elif args.mode == "nl2test":
        optim = NL(args.optpath, args.nlpath, args.nlidx)
        for opt in optim.get_opts():
            prompt = optim.get_prompt(template, opt)
            with open(outdir / f"{opt}.txt", "w") as f:
                f.write(prompt)
    elif args.mode == "template":
        if args.lib == "tf":
            optim_cls = SrcNLTest2TemplateTFLite
        elif args.lib == "tfxla":
            optim_cls = SrcNLTest2TemplateTFXLA
        elif args.lib == "torch":
            optim_cls = SrcNLTest2Template
        else:
            raise NotImplementedError
        optim = optim_cls(
            args.optpath, args.nlpath, args.testpath, args.nlidx, use_rl=True
        )
        num_of_prompts = 1
        for opt in optim.get_opts():
            prompts = optim.get_prompt(
                template, opt, num_of_prompts=num_of_prompts, num_model=args.num_model
            )
            if len(prompts) == 0:
                # No triggering inputs
                # use step1 prompts
                prompt = (Path(args.step1dir) / f"{opt}.txt").read_text()
                with open(outdir / f"{opt}.txt", "w") as f:
                    f.write(prompt)
            else:
                prompt = prompts[0]
                with open(outdir / f"{opt}.txt", "w") as f:
                    f.write(prompt)
        # Dump the updated tests, now containing the ids of selected examples.
        with open(args.testpath, "w") as f:
            json.dump(optim.tests, f, indent=4)
    elif args.mode == "template_nl":
        # optim = NLTest2Template(args.optpath, args.nlpath, args.testpath, args.nlidx, use_rl=True)
        optim = NLTest2Template(
            args.optpath, args.nlpath, args.testpath, args.nlidx, use_rl=use_rl
        )
        num_of_prompts = 1
        for opt in optim.get_opts():
            prompts = optim.get_prompt(
                template, opt, num_of_prompts=num_of_prompts, num_model=args.num_model
            )
            if len(prompts) == 0:
                # No triggering inputs
                # use step1 prompts
                prompt = (Path(args.step1dir) / f"{opt}.txt").read_text()
                with open(outdir / f"{opt}.txt", "w") as f:
                    f.write(prompt)
            else:
                prompt = prompts[0]
                with open(outdir / f"{opt}.txt", "w") as f:
                    f.write(prompt)
        # Dump the updated tests, now containing the ids of selected examples.
        with open(args.testpath, "w") as f:
            json.dump(optim.tests, f, indent=4)
    else:
        raise NotImplementedError
