from numpy.random import rand


def choose_from_list() -> bool:
    return rand() < 0.2


def change_tensor_dimension() -> bool:
    return rand() < 0.1


def add_tensor_dimension() -> bool:
    return rand() < 0.1


def rm_tensor_dimension() -> bool:
    return rand() < 0.05


def change_tensor_shape() -> bool:
    return rand() < 0.5


def change_tensor_layout() -> bool:
    return rand() < 0.1


def change_tensor_dtype() -> bool:
    return rand() < 0.3


def do_type_mutation() -> bool:
    return rand() < 0.2


def do_select_from_db() -> bool:
    return rand() < 0.2
