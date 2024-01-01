from enum import IntEnum, auto


class ResType(IntEnum):
    RANDOM = auto()
    STATUS = auto()
    VALUE = auto()
    REV_STATUS = auto()
    REV_VALUE = auto()
    FWD_STATUS = auto()
    FWD_VALUE = auto()
    REV_FWD_GRAD = auto()
    ND_GRAD = auto()
    PASS = auto()
    CRASH = auto()
    SKIP = auto()
    DIRECT_CRASH = auto()
    REV_CRASH = auto()
    FWD_CRASH = auto()
    ND_CRASH = auto()
    NAN = auto()
    ND_FAIL = auto()
    GRAD_NOT_COMPUTED = auto()
    FILTERED = auto()
    FWD_CANNOT = auto()
    JIT_CRASH = auto()
    JIT_STATUS = auto()
    JIT_VALUE = auto()
    JIT_FAIL = auto()
    REV_GRAD = auto()
    FWD_GRAD = auto()


class JitType(IntEnum):
    D_STATUS = auto()
    D_VALUE = auto()
    D_CRASH = auto()
    REV_STATUS = auto()
    REV_VALUE = auto()
    REV_CRASH = auto()
    REV_GRAD = auto()
    REV_GRAD_NAN = auto()
    FWD_STATUS = auto()
    FWD_VALUE = auto()
    FWD_CRASH = auto()
    FWD_GRAD = auto()
    FWD_GRAD_NAN = auto()
    FWD_GRAD_ZERO = auto()
    CRASH = auto()
    BOTH_CRASH = auto()
    D_TYPE_CRASH = auto()
    RANDOM = auto()
    PASS = auto()
    SKIP = auto()
    NAN = auto()


if __name__ == "__main__":
    a = [str(i).replace("ResType.", "") for i in ResType]
    print(", ".join(a))
