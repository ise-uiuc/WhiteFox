from enum import Enum


class ResultType(Enum):
    NOT_EQUIVALENT = 1
    SUCCESS = 2
    FAIL = 3
    BUG = 4
    ERROR = 5
    NOT_EQ_BK = 6
    NOT_EQ_GRAD = 7
    SKIP = 8
    BUG_NORMAL = 9
    NEQ_STATUS = 10
    NEQ_VALUE = 11
    BK_FAIL = 12


def count_results(results: list["ResultType"], keys: list[str] = []):
    """
    Count the number of BUG, FAIL and SUCCESS
    Return (#bug, #fail, #success)
    """
    res_type_count = {}
    for t in ResultType:
        res_type_count[str(t).replace("ResultType.", "")] = 0

    for result in results:
        res_type_count[str(result).replace("ResultType.", "")] += 1

    if len(keys) == 0:
        return res_type_count
    else:
        res = {}
        for k in keys:
            res[k] = res_type_count[k]
        return res


def comment_code(code):
    code_lines = code.split("\n")
    for i in range(len(code_lines)):
        code_lines[i] = "# " + code_lines[i]
    return "\n".join(code_lines) + "\n"
