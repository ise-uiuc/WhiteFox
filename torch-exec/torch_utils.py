import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from helper_torch import (
    DirectInv,
    RevInv,
    FwdInv,
    NDCheck,
    allow_error,
    is_crash,
    DirectJitInv,
    Filter,
    CopyInputs,
    get_trainable_fn,
    to_cuda,
)
from classes.torch_library import TorchLibrary
from constant.returntypes import ResType

TEST_FUNC_EXEC_NAME = "func"
IS_DEBUG_MODE = False
__EXEC = exec
__EVAL = eval

def print_debug(*args):
    if IS_DEBUG_MODE:
        print(*args)

def check_status(fn, inputs):
    # check the status when the inputs are high-precision
    status, _, _ = DirectInv(fn, CopyInputs(inputs, precise=True))
    return status == "success"

def validate(fn, inputs, device='cpu'):
    status, err, _ = DirectInv(fn, CopyInputs(inputs))
    if status == "success":
        return ResType.PASS, err
    else:
        return ResType.JIT_FAIL, err


def check_jit_value(fn, inputs, device):
    # check the value between direct and jit
    # you can specify the context and fn before calling this function
    # like fn.train(False) or torch.no_grad()

    errors = {}

    direct_status, direct_value, direct_err = DirectInv(fn, inputs, device=device)
    errors["direct"] = direct_err
    if is_crash(direct_err):
        return (ResType.DIRECT_CRASH, errors)

    is_random = False
    is_nan = False
    for _ in range(9):
        direct_status_, direct_value_, direct_err_ = DirectInv(fn, inputs, device=device)
        if direct_status != direct_status_ or not TorchLibrary.is_equal(
            direct_value, direct_value_, equal_nan=True
        ):
            is_random = True
            # return (ResType.RANDOM, errors)
        elif not TorchLibrary.is_equal(direct_value, direct_value_):
            is_nan = True
            # return (ResType.NAN, errors)
    
    jit_status, jit_value, jit_err, jit_fn = DirectJitInv(fn, inputs, mode="inductor", device=device)
    jit_restype = ResType.PASS
    errors["jit"] = jit_err

    if is_crash(jit_err):
        print_debug(jit_err)
        jit_restype = ResType.JIT_CRASH
    elif jit_status != direct_status and not allow_error(jit_err):
        print_debug(direct_status)
        print_debug(direct_err)
        print_debug(jit_status)
        print_debug(jit_err)
        jit_restype = ResType.JIT_STATUS
    elif jit_status == "fail":
        print_debug(direct_err)
        print_debug(jit_err)
        jit_restype = ResType.JIT_FAIL
    elif not TorchLibrary.is_equal(direct_value, jit_value):
        if is_random:
            jit_restype = ResType.RANDOM
        elif is_nan:
            jit_restype = ResType.NAN
        else:
            print_debug(direct_value)
            print_debug(jit_value)
            errors["direct"] = str(direct_value)
            errors["jit"] = str(jit_value)
            jit_restype = ResType.JIT_VALUE
    return jit_restype, errors, jit_fn

def check_grad_helper(fn, jit_fn, inputs, device, mode="rev"):
    if mode == "rev":
        Inv = RevInv
        crash = ResType.REV_CRASH
        status = ResType.REV_STATUS
        value = ResType.REV_VALUE
        grad = ResType.REV_GRAD
    elif mode == "fwd":
        Inv = FwdInv
        crash = ResType.FWD_CRASH
        status = ResType.FWD_STATUS
        value = ResType.FWD_VALUE
        grad = ResType.FWD_GRAD
    else:
        raise Exception("mode not supported")

    restype = ResType.PASS
    errors = {}
    _status, _value, _grad, _err = Inv(fn, inputs, device=device)
    _status_jit, _value_jit, _grad_jit, _err_jit = Inv(jit_fn, inputs, device=device)
    if is_crash(_err_jit):
        print_debug(_err_jit)
        restype = crash
        errors[f"jit-{mode}"] = _err_jit
    elif _status != _status_jit and not allow_error(_err_jit):
        print_debug(_err)
        print_debug(_err_jit)
        restype = status
        errors[f"jit-{mode}"] = _err_jit
    elif not TorchLibrary.is_equal(_value, _value_jit):
        print_debug(_value)
        print_debug(_value_jit)
        restype = value
    elif not TorchLibrary.is_equal(_grad, _grad_jit):
        print_debug(_grad)
        print_debug(_grad_jit)
        restype = grad

    return restype, errors
    



def check_jit_grad(fn, jit_fn, inputs, device):
    errors = {}
    jit_restype = ResType.PASS

    # Reverse mode AD
    rev_restype, rev_errors = check_grad_helper(fn, jit_fn, inputs, device, mode="rev")
    if rev_restype != ResType.PASS:
        jit_restype = rev_restype
        errors.update(rev_errors)
        return jit_restype, errors
    
    # Forward mode AD
    fwd_restype, fwd_errors = check_grad_helper(fn, jit_fn, inputs, device, mode="fwd")
    if fwd_restype != ResType.PASS:
        jit_restype = fwd_restype
        errors.update(fwd_errors)
        return jit_restype, errors

    return jit_restype, errors
    

def test_jit_oracle(fn, inputs, device='cpu', test_ad=False):
    inputs = CopyInputs(tuple(inputs), device=device)
    errors = {
        "direct": "",
        "jit": "",
    }

    if 'cuda' in device:
        to_cuda(fn, device)

    # you need to disable the train at the beginning
    with torch.no_grad():
        fn.train(False)
        jit_restype, jit_err, _ = check_jit_value(fn, inputs, device=device)
        errors.update(jit_err)
    
    if test_ad and jit_restype == ResType.PASS:
        fn.train(True)
        # check the value first
        new_jit_restype, new_jit_err, jit_fn = check_jit_value(fn, inputs, device=device)
        if new_jit_restype == ResType.PASS:
            # then check the gradient
            jit_restype, errors = check_jit_grad(fn, jit_fn, inputs, device=device)

    return (jit_restype, errors)



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def set_state():
    # refresh the global state
    torch.set_grad_enabled(True)
    torch.set_anomaly_enabled(False)
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type(torch.FloatTensor)


def test_wrapper(func_def_code, rand_seed, grad_tensors, device, test_fn="test_jit_oracle", test_ad=False):
    errors = {}
    if len(grad_tensors):
        set_seed(rand_seed)
        set_state()
        try:
            __EXEC(func_def_code, globals())
        except Exception as e:
            ret = ResType.SKIP
            print(e)
            print(func_def_code)
        else:
            try:
                inputs_str = f"({', '.join(grad_tensors)},)"

                # no grad env
                if test_fn == "test_jit_oracle":
                    ret, errors = __EVAL(f"{test_fn}(func, {inputs_str}, '{device}', {test_ad})")
                else: 
                    ret, errors = __EVAL(f"{test_fn}(func, {inputs_str}, '{device}')")

            except Exception as _run_error:
                print(_run_error)
                errors['crash'] = str(_run_error)
                ret = ResType.CRASH
    else:
        ret = ResType.SKIP
        print("NO INPUT")

    return (ret, errors)
