class API:
    def __init__(self, api_name):
        self.api = api_name

    def mutate(self):
        pass

    def to_code(self) -> str:
        pass

    def to_dict(self) -> dict:
        pass

    @staticmethod
    def generate_args_from_record(record: dict) -> dict:
        pass

    @staticmethod
    def indent_code(code, indent_times=1):
        if code == "":
            return ""
        codes = code.split("\n")
        result = []
        for code in codes:
            if code == "":
                continue
            result.append("    " * indent_times + code)
        return "\n".join(result) + "\n"

    @staticmethod
    def try_except_code(test_code, error_res, else_code="", error_code=""):
        code = ""
        code += "try:\n"
        code += API.indent_code(test_code)
        code += "except Exception as e:\n"
        code += API.indent_code(f"{error_res} = str(e)\n")
        if error_code != "":
            code += API.indent_code(error_code)
        if else_code != "":
            code += "else:\n"
            code += API.indent_code(else_code)
        return code
