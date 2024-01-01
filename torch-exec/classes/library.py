from classes.argument import *
from classes.api import *
from os.path import join
import os


class Library:
    def __init__(self, directory) -> None:
        def init_dir(dir_name):
            os.makedirs(join(dir_name, "success"), exist_ok=True)
            os.makedirs(join(dir_name, "potential-bug"), exist_ok=True)
            os.makedirs(join(dir_name, "fail"), exist_ok=True)

        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.output = {
            OracleType.CRASH: join(directory, "crash-oracle"),
            OracleType.CUDA: join(directory, "cuda-oracle"),
            OracleType.PRECISION: join(directory, "precision-oracle"),
        }
        for dir_name in self.output.values():
            init_dir(dir_name)

    @staticmethod
    def generate_code():
        pass

    @staticmethod
    def write_to_dir(dir, code, enable_output: bool = True) -> str:
        if not enable_output:
            return

        filenames = os.listdir(dir)
        max_name = 0
        for name in filenames:
            if name.endswith(".py") and not name.startswith("diff-"):
                max_name = max(max_name, int(name.replace(".py", "")))
        new_name = str(max_name + 1)
        with open(join(dir, new_name + ".py"), "w") as f:
            f.write(code)
        return new_name
