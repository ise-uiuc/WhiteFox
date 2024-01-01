import os

from gen_prompt import gen_prompt_from_template
from utils import (
    copy_file,
    create_file,
    extract_function_body,
    remove_commented_lines,
    simplify_func,
    remove_extra_blank_lines,
)


def extract_source_from_llvm(data: dict):
    """
    extract the origin source code and example IR files to `source-code-data/llvm/llvm-lib`
    you can invoke the `extract_file_from_llvm` to get the `source-code-data/llvm/llvm-lib` collected
    """
    # begin traversal
    for passname, pass_infos in data.items():
        for index, pass_info in enumerate(pass_infos["hints"]):
            assert pass_info["type"] == "trigger", "type must be trigger now"

            dest_file1 = copy_file(
                pass_info["codes"][0], "source-code-data/llvm/llvm-lib"
            )
            if len(pass_info["codes"]) == 2:
                dest_file2 = copy_file(
                    pass_info["codes"][1], "source-code-data/llvm/llvm-lib"
                )
            print(f"codes have been saved in {dest_file1}")
            example_file = copy_file(
                pass_info["examples"][0], "source-code-data/llvm/llvm-lib"
            )  # TODO Only first cpp file!
            print(f"example file have been saved in {example_file}")


def extract_func_body_from_source(data: dict):
    """
    To get the suitable funcbody for every target line in every pass,
    the extracted func_body file will be in `source-code-data/llvm/llvm-func-body`.
    """
    # collect the dict of {func:[targetlines]} in advance
    target_dict = dict()
    for passname, pass_infos in data.items():
        for index, pass_info in enumerate(pass_infos["hints"]):
            if pass_info["func"][-1] in target_dict:
                target_dict[pass_info["func"][-1]].append(pass_info["target_line"])
            else:
                target_dict[pass_info["func"][-1]] = [pass_info["target_line"]]

    # begin traversal
    for passname, pass_infos in data.items():
        for index, pass_info in enumerate(pass_infos["hints"]):
            dest_file1 = os.path.join(
                "source-code-data/llvm/llvm-lib",
                os.path.basename(pass_info["codes"][0]),
            )
            if len(pass_info["codes"]) == 2:
                dest_file2 = os.path.join(
                    "source-code-data/llvm/llvm-lib",
                    os.path.basename(pass_info["codes"][1]),
                )

            func_body_folder = "source-code-data/llvm/llvm-func-body"
            func_body_folder_file = os.path.join(
                func_body_folder, passname + f"{index}.cpp"
            )
            if not os.path.exists(func_body_folder_file):
                create_file(func_body_folder_file)

            func_list = pass_info["func"]
            func_body_list = []
            # for func in func_list:
            # 	print(f"adding func {func}")
            func_body_list.append(extract_function_body(dest_file1, func_list[0]))
            if len(func_list) > 1:
                if len(pass_info["codes"]) == 2:
                    func_body_list.append(
                        extract_function_body(dest_file2, func_list[1])
                    )
                else:
                    func_body_list.append(
                        extract_function_body(dest_file1, func_list[1])
                    )

            # print("\n".join(func_body_list))

            # target line
            target_line = pass_info["target_line"]

            func_body_list_after = simplify_func(
                func_list, func_body_list, target_line, target_dict
            )
            with open(func_body_folder_file, "w") as body_file:
                for func_body in func_body_list_after:
                    if func_body is not None:
                        body_file.write(func_body)
                        body_file.write("\n")

            remove_extra_blank_lines(func_body_folder_file)


def gen_prompt(data: dict, file_string="ll", gen_dir_path=None, examples=None):
    """
    According to the template.md and files in source-code-data/llvm/llvm-func-body,
    generate the corresponding prompt file(in `source-code-data/llvm/llvm-gen-prompt`)
    """

    # begin traversal
    for passname, pass_infos in data.items():
        for index, pass_info in enumerate(pass_infos["hints"]):
            func_list = pass_info["func"]
            example_file = os.path.join(
                "source-code-data/llvm/llvm-lib",
                os.path.basename(pass_info["examples"][0]),
            )
            example_ll = ""
            target_line = pass_info["target_line"]
            func_body_folder = "source-code-data/llvm/llvm-func-body"
            func_body_folder_file = os.path.join(
                func_body_folder, passname + f"{index}.cpp"
            )
            func_body_list = []
            # with open(func_body_folder_file, "r") as file:
            #     func_body_list = [file.read()]
            gen_prompt_from_template(
                func_list,
                func_body_list,
                example_ll,
                passname,
                target_line,
                index,
                file_string,
                gen_dir_path,
                examples,
            )


def traversal(data: dict):
    # collect the dict of {func:[targetlines]} in advance
    target_dict = dict()
    for passname, pass_infos in data.items():
        for index, pass_info in enumerate(pass_infos["hints"]):
            if pass_info["func"][-1] in target_dict:
                target_dict[pass_info["func"][-1]].append(pass_info["target_line"])
            else:
                target_dict[pass_info["func"][-1]] = [pass_info["target_line"]]

    # begin traversal
    for passname, pass_infos in data.items():
        for index, pass_info in enumerate(pass_infos["hints"]):
            assert pass_info["type"] == "trigger", "type must be trigger now"

            # if not os.path.exists(pass_info["codes"][0]):
            dest_file1 = copy_file(
                pass_info["codes"][0], "source-code-data/llvm/llvm-lib"
            )
            if len(pass_info["codes"]) == 2:
                dest_file2 = copy_file(
                    pass_info["codes"][1], "source-code-data/llvm/llvm-lib"
                )
            print(f"codes have been saved in {dest_file1}")
            example_file = copy_file(
                pass_info["examples"][0], "source-code-data/llvm/llvm-lib"
            )  # TODO Only first cpp file!
            print(f"example file have been saved in {example_file}")
            # with open(example_file, "r") as file:
            # example_ll = file.read()
            example_ll = remove_commented_lines(example_file)

            func_body_folder = "source-code-data/llvm/llvm-func-body"
            func_body_folder_file = os.path.join(
                func_body_folder, passname + f"{index}.cpp"
            )
            if not os.path.exists(func_body_folder_file):
                create_file(func_body_folder_file)

            func_list = pass_info["func"]
            func_body_list = []
            # for func in func_list:
            # 	print(f"adding func {func}")
            func_body_list.append(extract_function_body(dest_file1, func_list[0]))
            if len(func_list) > 1:
                if len(pass_info["codes"]) == 2:
                    func_body_list.append(
                        extract_function_body(dest_file2, func_list[1])
                    )
                else:
                    func_body_list.append(
                        extract_function_body(dest_file1, func_list[1])
                    )

            # print("\n".join(func_body_list))

            # target line
            target_line = pass_info["target_line"]
            # print(f"targetline now: {target_line}")

            func_body_list_after = simplify_func(
                func_list, func_body_list, target_line, target_dict
            )
            with open(func_body_folder_file, "w") as body_file:
                for func_body in func_body_list_after:
                    if func_body is not None:
                        body_file.write(func_body)
                        body_file.write("\n")

            # content = remove_commented_lines(func_body_folder_file, "/")
            # with open(func_body_folder_file, "w") as body_file:
            # 	body_file.write(content)

            remove_extra_blank_lines(func_body_folder_file)

            # gen_prompt(func_list, func_body_list_after, example_ll, passname, target_line, index)
            gen_prompt_from_template(
                func_list,
                func_body_list_after,
                example_ll,
                passname,
                target_line,
                index,
            )


"""
Codes Below are only for debug!
"""

if __name__ == "__main__":
    import json

    with open("example.json", "r") as file:
        data = json.load(file)
    extract_source_from_llvm(data)
    extract_func_body_from_source(data)
    gen_prompt(data)
    # traversal(data)
