import torch
import json
from gradcheck import (
    get_numerical_jacobian,
    _as_tuple,
    _differentiable_outputs,
    get_jacobians,
)
import random
from torch.overrides import is_tensor_like
from torch.autograd import forward_ad as fwad
from torch.autograd.functional import jacobian
from torch.func import jacfwd, jacrev

from classes.torch_library import TorchLibrary

NEIGHBOR_NUM = 5
NEIGHBOR_STEP = 1e-4
EPS = 1e-6
ATOL = 1e-1
RTOL = 1e-3

USE_NEW_GRAD_FUNC = True



def is_backward_tensor(res):
    if isinstance(res, torch.Tensor) and (res.grad_fn or res.requires_grad):
        return True
    else:
        return False


def is_backward_res(res):
    if is_backward_tensor(res):
        return True
    else:
        if isinstance(res, (tuple, list)):
            flag = False
            for x in res:
                if is_backward_tensor(x):
                    flag = True
            return flag
        else:
            return False

def get_trainable_fn(res, fn):
    if isinstance(res, torch.Tensor):
        return [fn]
    elif isinstance(res, (tuple, list)):
        fn_list = []
        for idx, x in enumerate(res):
            fn_list += get_trainable_fn(x, (lambda *inp, idx=idx: (fn(*inp)[idx])))
        return fn_list
    else:
        return []


def backward(res):
    def backward_tensor_with_grad(res):
        if isinstance(res, torch.Tensor) and res.grad_fn:
            res.sum().backward()
            return True
        else:
            return False

    if not backward_tensor_with_grad(res) and isinstance(res, (tuple, list)):
        for x in res:
            if backward_tensor_with_grad(x):
                break


def get_res_info(res):
    if is_backward_tensor(res):
        return json.dumps(
            {"shape": [int(i) for i in res.size()], "dtype": str(res.dtype)}
        )
    elif isinstance(res, (tuple, list)):
        res = []
        for x in res:
            res.append(get_res_info(x))
        return str(res)
    else:
        return "null"


def is_equal_tensor(x, y):
    assert torch.is_tensor(x), "not a tensor"
    assert torch.is_tensor(y), "not a tensor"
    assert x.size() == y.size(), "size is not equal"

    max_atol = 0.0
    max_rtol = 0.0
    max_aidx = None
    max_ridx = None
    from itertools import product

    for idx in product(*[range(m) for m in x.size()]):
        diff = abs(x[idx] - y[idx])
        if diff > max_atol:
            max_aidx = idx
            max_atol = diff
            print(x[idx])
            print(y[idx])
            print(torch.allclose(x[idx], y[idx], rtol=1e-3, atol=1e-1))
        rtol = diff / max(x[idx], y[idx])
        if rtol > max_rtol:
            max_ridx = idx
            max_rtol = rtol
        if diff > 1e-1 + 1e-3 * abs(y[idx]):
            print("Not equal " + str(idx))
            print(x[idx])
            print(y[idx])
    return max_atol, max_rtol, max_aidx, max_ridx


def has_nan(x):
    if isinstance(x, torch.Tensor):
        return bool(torch.any(torch.isnan(x)))
    elif isinstance(x, (list, tuple)):
        flag = False
        for i in x:
            flag = flag or has_nan(i)
        return flag
    else:
        return False


def all_zero(x):
    if isinstance(x, torch.Tensor):
        return torch.count_nonzero(x) == 0
    elif isinstance(x, (list, tuple)):
        flag = True
        for i in x:
            flag = flag and all_zero(i)
        return flag
    else:
        return False


def is_differentiable(
    fn, inputs, neighbor_step=NEIGHBOR_STEP, eps=EPS, atol=ATOL, rtol=RTOL
):
    def get_neighbor(inputs, type="left"):
        res = []
        for tensor in inputs:
            if type == "left":
                temp = tensor.clone() - neighbor_step
            elif type == "right":
                temp = tensor.clone() + neighbor_step
            else:
                # sign = torch.sign(torch.randn(tensor.shape))
                # temp = tensor.clone() + sign * neighbor_step
                # TODO: should we consider the dtype issue?
                delta = torch.empty(tensor.shape)
                torch.nn.init.uniform_(delta, -neighbor_step, neighbor_step)
                temp = tensor.clone() + delta
            res.append(temp)
        return tuple(res)

    inputs = _as_tuple(CopyInputs(inputs, grad=True))
    out = fn(*CopyInputs(inputs, grad=True))
    jacobians = get_numerical_jacobian(fn, inputs, eps=eps)
    for _ in range(NEIGHBOR_NUM):
        inputs_rand = get_neighbor(inputs, type="rand")

        rand_out = fn(*CopyInputs(inputs_rand, grad=True))
        if not TorchLibrary.is_equal(out, rand_out, atol=atol, rtol=rtol):
            return False

        rand_jacobian = get_numerical_jacobian(fn, inputs_rand, eps=eps)
        for i in range(len(jacobians)):
            for j in range(len(jacobians[i])):
                if not torch.allclose(
                    jacobians[i][j], rand_jacobian[i][j], atol, rtol
                ):
                    return False
    return True


def is_grad_differentiable(
    func, inputs, neighbor_step=NEIGHBOR_STEP, eps=EPS, atol=ATOL, rtol=RTOL
):
    if not is_differentiable(
        func,
        inputs,
        neighbor_step=neighbor_step,
        eps=eps,
        atol=atol,
        rtol=rtol,
    ):
        return False

    tupled_inputs = _as_tuple(inputs)
    outputs = _as_tuple(func(*tupled_inputs))
    tupled_grad_outputs = tuple(
        torch.testing.make_tensor(
            x.shape,
            dtype=x.dtype
            if x.is_floating_point() or x.is_complex()
            else torch.double,
            device=x.device,
            low=-1,
            high=1,
            requires_grad=True,
        )
        for x in outputs
    )

    num_outputs = len(tupled_grad_outputs)

    # NB: We need to save the requires_grad information about the inputs here because gradcheck detaches inputs
    #     before running forward mode AD
    diff_input_args_indices = set(
        i
        for i, x in enumerate(tupled_inputs)
        if is_tensor_like(x) and x.requires_grad
    )
    diff_grad_output_indices = set(
        i for i, x in enumerate(tupled_grad_outputs) if x.requires_grad
    )

    def new_func(*args):
        # Restore the requires_grad information
        input_args = tuple(
            x.requires_grad_() if i in diff_input_args_indices else x
            for i, x in enumerate(args[:-num_outputs])
        )
        outputs = _differentiable_outputs(func(*input_args))
        grad_outputs = tuple(
            x.requires_grad_() if i in diff_grad_output_indices else x
            for i, x in enumerate(args[-num_outputs:])
        )
        diff_input_args = tuple(
            x for i, x in enumerate(input_args) if i in diff_input_args_indices
        )
        grad_inputs = torch.autograd.grad(
            outputs,
            diff_input_args,
            grad_outputs,
            create_graph=True,
            allow_unused=True,
        )
        grad_inputs = tuple(g for g in grad_inputs if g is not None)
        return grad_inputs

    return is_differentiable(
        new_func,
        tupled_inputs + tupled_grad_outputs,
        neighbor_step=neighbor_step,
        eps=eps,
        atol=atol,
        rtol=rtol,
    )


def allow_error(err):
    _allow_errors = [
        "a leaf Variable that requires grad is being used in an in-place operation",
        "support automatic differentiation",
        "Can't export tensors that require gradient, use tensor.detach()",
        "is not differentiable with respect to argument",
        "not implemented for",
        "is not implemented",
        "is not differentiable",
        "does not support",
        "has no attribute 'requires_grad'",
        # "must be Tensor, not bool",
        "must be Number, not Tensor",
        "argument 'dual' (position 1) must be Tensor",
        "value cannot be converted",
        "cannot resize variables that require grad",
        "is only supported for complex tensors",
        "not supported",
        "Trying to set a forward gradient that has a different size",
        "For complex Tensors, both grad_output and output are required to have the same dtype",
        "Numerical gradient for function expected to be zero", # for numerical gradient check
        "Can't detach views in-place", # for detach in forward mode AD
        "Can't call numpy() on Tensor that requires grad.",
        "This could be because the operator doesn't exist for this backend", # no operator for backend
        "vmap: We do not yet support calling random operations inside of vmap. Please perform random operations outside of vmap as a workaround", # for forward mode AD
        "RuntimeError: mixed dtype (CPU)", # for gradcheck, since we use float64 to check
        "stack expects a non-empty TensorList", # for reverse mode AD jacobian computation
        "Can't call apply_() on Variable that requires grad. Use var.detach().apply_() instead.", 
        "modified by an inplace operation",
        "can't allocate memory",
        "You must implement the jvp function for custom autograd.Function to use it with forward mode AD.",
        "received an invalid combination of arguments",
        "out of memory",
    ]
    if USE_NEW_GRAD_FUNC:
        _allow_errors += [
            "All outputs of f must be floating-point or complex Tensors, got Tensor with dtype",
            "called random operation while in randomness error mode.",
            "must have a setup_context",
            "is unsupported",
            "not yet support",
            "Cannot access data pointer of Tensor that doesn't have storage", # not sure whether it's a bug
            "vmap: It looks like you're calling .item() on a Tensor",
            "Expected all inputs to be real but received complex",
            "grad can be implicitly created only for real scalar",
            "Expected all outputs to be real but"
        ]
    for a_err in _allow_errors:
        if a_err in err:
            return True
    return False


def is_crash(err):
    return "INTERNAL ASSERT FAILED" in err


def CopyInputs(
    inputs: list[torch.Tensor], grad=False, device="cpu", precise=False
):
    new_inputs = []
    to_kwargs = {"device": device}
    for inp in inputs:
        if not isinstance(inp, torch.Tensor):
            new_inputs += [inp]
            continue

        if precise:
            if inp.is_floating_point():
                to_kwargs["dtype"] = torch.float64
            else:
                to_kwargs["dtype"] = torch.complex128

        if grad:
            new_inputs += [inp.clone().to(**to_kwargs).requires_grad_()]
        else:
            new_inputs += [inp.clone().to(**to_kwargs)]
    return tuple(new_inputs)


def DirectInv(fn, inputs, device="cpu"):
    value = None
    err_msg = ""
    new_inputs = CopyInputs(inputs, device=device)
    try:
        value = fn(*new_inputs)
    except Exception as e:
        status = "fail"
        err_msg = str(e)
    else:
        status = "success"
    return status, value, err_msg


def RevInv(fn, inputs, device="cpu"):
    value = None
    err_msg = ""
    gradient = None
    try:
        inputs_0 = CopyInputs(inputs, grad=True, device=device)
        value = fn(*inputs_0)
        backward(value)
        # gradient = get_jacobians(fn, inputs_1, backward_ad=True)
        fn_list = get_trainable_fn(value, fn)
        if len(fn_list):
            gradient = []
            for new_fn in fn_list:
                inputs_1 = CopyInputs(inputs, grad=True, device=device)
                if USE_NEW_GRAD_FUNC:
                    gradient.append(jacrev(new_fn)(*inputs_1))
                else:
                    gradient.append(jacobian(new_fn, inputs_1))
    except Exception as e:
        status = "fail"
        err_msg = str(e)
    else:
        status = "success"
    return status, value, gradient, err_msg


def FwdInv(fn, inputs, device="cpu"):
    value = None
    err_msg = ""
    gradient = None
    try:
        inputs_0 = CopyInputs(inputs, grad=True, device=device)
        with fwad.dual_level():
            dual_inputs = []
            for inp in inputs_0:
                tan = torch.rand_like(inp)
                dual_inputs += [fwad.make_dual(inp, tan)]
            out = fn(*dual_inputs)
            if isinstance(out, tuple):
                value = []
                for o in out:
                    temp, _ = fwad.unpack_dual(o)
                    value.append(o)
                value = tuple(value)
            else:
                value, _ = fwad.unpack_dual(out)
        # gradient = get_jacobians(fn, inputs_1, backward_ad=False)
        fn_list = get_trainable_fn(value, fn)
        if len(fn_list):
            gradient = []
            for new_fn in fn_list:
                inputs_1 = CopyInputs(inputs, grad=True, device=device)
                if USE_NEW_GRAD_FUNC:
                    gradient.append(jacfwd(new_fn)(*inputs_1))
                else:
                    gradient.append(jacobian(new_fn, inputs_1, vectorize=True, strategy="forward-mode"))
    except Exception as e:
        status = "fail"
        err_msg = str(e)
    else:
        status = "success"
    return status, value, gradient, err_msg


def NDCheck(fn, inputs, mode="rev", device="cpu"):
    err_msg = ""
    try:
        inputs = CopyInputs(inputs, grad=True, device=device, precise=True)
        kwargs = {
            "eps": EPS,
            "atol": ATOL,
            "rtol": RTOL,
            "check_forward_ad": mode == "fwd",
            "check_backward_ad": mode == "rev",
            "check_sparse_nnz": False,
        }
        for inp in inputs:
            if inp.is_sparse:
                kwargs["check_sparse_nnz"] = True
                break
        value = fn(*CopyInputs(inputs, device=device))
        fn_list = get_trainable_fn(value, fn)
        for new_fn in fn_list:
            inputs = CopyInputs(inputs, grad=True, device=device, precise=True)
            torch.autograd.gradcheck(new_fn, inputs, **kwargs)
    except Exception as e:
        status = "fail"
        err_msg = str(e)
    else:
        status = "success"
    return status, err_msg


def Grad(fn, inputs, device="cpu"):
    new_inputs = CopyInputs(inputs, grad=True, device=device)
    outputs = _as_tuple(fn(*new_inputs))
    tupled_grad_outputs = tuple(
        torch.testing.make_tensor(
            x.shape,
            dtype=x.dtype
            if x.is_floating_point() or x.is_complex()
            else torch.double,
            device=x.device,
            low=1,
            high=1,
            requires_grad=True,
        )
        for x in outputs
    )
    num_outputs = len(tupled_grad_outputs)

    diff_input_args_indices = set(
        i
        for i, x in enumerate(new_inputs)
        if is_tensor_like(x) and x.requires_grad
    )
    diff_grad_output_indices = set(
        i for i, x in enumerate(tupled_grad_outputs) if x.requires_grad
    )

    def new_func(*args):
        # Restore the requires_grad information
        args = args + tupled_grad_outputs
        input_args = tuple(
            x.requires_grad_() if i in diff_input_args_indices else x
            for i, x in enumerate(args[:-num_outputs])
        )
        # print("in func args: " + str(args))
        # print("in func input: " + str(input_args))
        outputs = _differentiable_outputs(fn(*input_args))
        grad_outputs = tuple(
            x.requires_grad_() if i in diff_grad_output_indices else x
            for i, x in enumerate(args[-num_outputs:])
        )
        diff_input_args = tuple(
            x for i, x in enumerate(input_args) if i in diff_input_args_indices
        )
        grad_inputs = torch.autograd.grad(
            outputs,
            diff_input_args,
            grad_outputs,
            create_graph=True,
            allow_unused=True,
        )
        grad_inputs = tuple(g for g in grad_inputs if g is not None)
        return grad_inputs

    return new_func


dtype_precision_dict = {
    "torch.bfloat16": 1,
    "torch.float16": 2,
    "torch.float32": 3,
    "torch.complex64": 3,
    "torch.float64": 4,
    "torch.complex128": 4,
}


def dtype_precision(dtype):
    dtype = str(dtype)
    if "int" in dtype or "bool" in dtype:
        return 0
    elif dtype in dtype_precision_dict.keys():
        return dtype_precision_dict[dtype]
    else:
        assert 0, f"No such dtype: {dtype}"


def is_high_to_low_precision(inputs: list[torch.Tensor], output):
    def _check(output: torch.Tensor):
        for input in inputs:
            if dtype_precision(input.dtype) > dtype_precision(output.dtype):
                return True
        return False

    if isinstance(output, (tuple, list)):
        for o in output:
            if _check(o):
                return True
        return False
    elif isinstance(output, torch.Tensor):
        return _check(output)
    else:
        return False


def Filter(fn, inputs, is_ND=False) -> str:
    def is_nan_grad(grad: list[torch.Tensor]):
        ret = False
        if isinstance(grad, (list, tuple)):
            ret = ret or any(is_nan_grad(t) for t in grad)
        elif isinstance(grad, torch.Tensor):
            ret = ret or torch.any(torch.isnan(grad))
        return ret

    # NaN Check
    _, rev_value, rev_grad, _ = RevInv(fn, inputs)
    _, fwd_value, fwd_grad, _ = FwdInv(fn, inputs)
    has_rev_grad = rev_grad is not None
    has_fwd_grad = fwd_grad is not None
    if has_rev_grad:
        rev_nan = is_nan_grad(rev_grad)
    if has_fwd_grad:
        fwd_nan = is_nan_grad(fwd_grad)
    if (not has_rev_grad or rev_nan) and (not has_fwd_grad or fwd_nan):
        return "NaN"
    elif has_rev_grad and has_fwd_grad and rev_nan != fwd_nan:
        # print(rev_nan)
        # print(fwd_nan)
        return "NaN-error"

    # Presicion
    if is_ND:
        inputs = CopyInputs(inputs, precise=True)
        _, output_value, _ = DirectInv(fn, inputs)
    else:
        output_value = rev_value if rev_value is not None else fwd_value
    # if is_high_to_low_precision(inputs, output_value):
    #     return "Precision"

    # Differentiability
    # new_inputs = CopyInputs(inputs, grad=True, precise=True)
    if not is_differentiable(fn, inputs):
        return "Non-Diff"
    else:
        return "Pass"


def Jit(fn):
    return torch.jit.script(fn)


def Trace(fn, inputs, device="cpu"):
    new_inputs = CopyInputs(inputs, device=device)
    return torch.jit.trace(fn, new_inputs)


def Script(fn):
    return torch.jit.script(fn)

def Inductor(fn):
    return torch.compile(fn)

def DirectJitInv(fn, inputs, device="cpu", mode="trace"):
    value = None
    jit_fn = None
    err_msg = ""
    new_inputs = CopyInputs(inputs, device=device)
    try:
        if mode == "inductor":
            jit_fn = torch.compile(fn)
        elif mode == "trace":
            jit_fn = Trace(fn, new_inputs, device=device)
        else:
            jit_fn = Script(fn)
        value = jit_fn(*new_inputs)
    except Exception as e:
        status = "fail"
        err_msg = str(e)
    else:
        status = "success"
    return status, value, err_msg, jit_fn


def allow_jit_error(err):
    _allow_jit_errs = [
        "Only tensors, lists, tuples of tensors, or dictionary of tensors can be output from traced functions",
    ]
    for e in _allow_jit_errs:
        if e in err:
            return True
    return False


def is_inplace_grad_err(err):
    _inplace_grad_errs = [
        "modified by an inplace operation",
        "modified inplace",
    ]
    for e in _inplace_grad_errs:
        if e in err:
            return True
    return False


def to_cuda(model: torch.nn.Module, device: str = "cuda"):
    for attr in dir(model):
        if isinstance(getattr(model, attr), torch.Tensor):
            setattr(model, attr, getattr(model, attr).to(device))