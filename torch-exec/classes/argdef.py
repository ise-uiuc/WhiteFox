from classes.argument import Argument


class ArgDef:
    def __init__(self):
        self.name: str = ""
        self.is_optional: bool = False
        self.must_use_name: bool = False
        self.type: set = set()
        self.default_value: str = ""
        self.description: str = ""
        self.case: Argument = None
        self.record = None
        self.ignore: bool = False
        self.disable = False

    @staticmethod
    def new(record):
        arg = ArgDef()
        arg.name = record["name"]
        arg.is_optional = record["is_optional"]
        arg.must_use_name = record["must_use_name"]
        arg.type = set(record["type"])
        arg.default_value = record["default_value"]
        arg.description = record["description"]
        return arg
