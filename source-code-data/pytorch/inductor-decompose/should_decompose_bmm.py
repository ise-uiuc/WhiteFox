MIN_FIRST_DIMENSION_DECOMPOSITION = 10240
MAX_OTHER_DIMENSION_DECOMPOSITION = 32

def should_decompose_common(
    mat1: Tensor, mat2: Tensor, input: Optional[Tensor] = None
) -> bool:
    return (
        torch._inductor.config.decompose_mem_bound_mm
        and check_device(mat1, mat2)
        and not utils.any_is_symbolic(mat1, mat2, input)
    )


def should_decompose_bmm(mat1, mat2) -> bool:
    mat1 = mat1.meta["val"]
    mat2 = mat2.meta["val"]
    if not should_decompose_common(mat1, mat2):
        return False
    else:
        if len(mat1.shape) != 3 or len(mat2.shape) != 3:
            return False
        if mat1.shape[0] < MIN_FIRST_DIMENSION_DECOMPOSITION:
            return False
        # 2 of m, n, k must be <= MAX_OTHER_DIMENSION_DECOMPOSITION
        if (mat1.shape[1] < MAX_OTHER_DIMENSION_DECOMPOSITION) + (
            mat1.shape[2] < MAX_OTHER_DIMENSION_DECOMPOSITION
        ) + (mat2.shape[2] < MAX_OTHER_DIMENSION_DECOMPOSITION) < 2:
            return False
    return True