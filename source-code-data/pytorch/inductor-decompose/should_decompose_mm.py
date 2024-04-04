MIN_FIRST_DIMENSION_DECOMPOSITION = 10240
MAX_OTHER_DIMENSION_DECOMPOSITION = 32

def should_decompose_mm(mat1, mat2) -> bool:
    mat1 = mat1.meta["val"]
    mat2 = mat2.meta["val"]
    return (
        should_decompose_common(mat1, mat2)
        and len(mat1.shape) == 2
        and len(mat2.shape) == 2
        and mat1.shape[0] >= MIN_FIRST_DIMENSION_DECOMPOSITION
        and mat2.shape[0] < MAX_OTHER_DIMENSION_DECOMPOSITION
        and mat2.shape[1] < MAX_OTHER_DIMENSION_DECOMPOSITION
    )

def should_decompose_common(
    mat1: Tensor, mat2: Tensor, input: Optional[Tensor] = None
) -> bool:
    return (
        torch._inductor.config.decompose_mem_bound_mm
        and check_device(mat1, mat2)
        and not utils.any_is_symbolic(mat1, mat2, input)
    )