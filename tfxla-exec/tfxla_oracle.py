import tensorflow as tf
import numpy as np
from pathlib import Path
import argparse
from enum import IntEnum, auto
from collections import Counter
import os
import random

from tfxla_code_process import process_code
import run_nnsmith

class ResType(IntEnum):
    NaiveFail = auto()
    XLAFail = auto()
    ACFail = auto()
    Naive_XLAFail = auto()
    Naive_ACFail = auto()
    XLA_ACFail = auto()
    AllFail = auto() #PASS
    AllPass = auto() #PASS
    XLA_ACDiff = auto()
    Naive_ACDiff = auto()
    NaiveXLADiff = auto()
    AllDiff = auto()
    AllDiff_Rand = auto()
    AllDiff_LessLikely = auto()
    AllDiff_TypeMismatch = auto()

class DataType(IntEnum):
    Float = auto()
    Bool = auto()
    Int = auto()
    Str = auto()
    Null = auto()
    Tuple = auto()
    List = auto()
    TFTensor = auto()
    KerasTensor = auto()
    Unknown = auto()

FAIL_MAPPING = {"[1, 0, 0]":ResType.NaiveFail, "[0, 1, 0]":ResType.XLAFail, "[0, 0, 1]":ResType.ACFail,
                "[1, 1, 0]":ResType.Naive_XLAFail, "[1, 0, 1]":ResType.Naive_ACFail, "[0, 1, 1]":ResType.XLA_ACFail,
                "[1, 1, 1]":ResType.AllFail, "[0, 0, 0]":ResType.AllPass,
            }

NUM_MAPPING = {"[1, 0, 0]":ResType.XLA_ACDiff, "[0, 1, 0]":ResType.Naive_ACDiff, "[0, 0, 1]":ResType.XLA_ACDiff,
               "[0, 0, 0]":ResType.AllDiff
            }

TFXLA_TRIGGER_LOG_PATH = "/tmp/xla_trigger.log"

RANDOM_SEED = 42

def add_decorator(code: str, decorator: str) -> str:
    if "    def call" in code:
        # The indentation is 4 spaces
        code = code.replace("    def call", f"    {decorator}\n    def call")
    else:
        # The indentation is 2 spaces
        code = code.replace("  def call", f"  {decorator}\n  def call")
    return code

def get_type(output_data):
    """Get the type of the output data for comparison."""
    if output_data is None:
        return DataType.Null
    elif isinstance(output_data, bool):
        return DataType.Bool
    elif isinstance(output_data, int):
        return DataType.Int
    elif isinstance(output_data, str):
        return DataType.Str
    elif isinstance(output_data, float):
        return DataType.Float
    elif isinstance(output_data, tuple):
        return DataType.Tuple
    elif isinstance(output_data, list):
        return DataType.List
    elif isinstance(output_data, tf.Tensor):
        return DataType.TFTensor
    elif tf.keras.backend.is_keras_tensor(output_data):
        return DataType.KerasTensor
    else:
        return DataType.Unknown

def is_equal(x, y):
    x_type, y_type = get_type(x), get_type(y)
    if x_type != y_type and not (x_type in [DataType.List, DataType.Tuple] and y_type in [DataType.List, DataType.Tuple]):
        try:
            equal = np.allclose(np.array(x), np.array(y), atol=1e-02, equal_nan=True)
            return equal, "Value mismatch: {} vs {}".format(x, y)
        except:
            return False, "Type mismatch: {} vs {}".format(str(x_type), str(y_type))
    
    # The type that can be compared directly.
    if x_type in [DataType.Int, DataType.Bool, DataType.Null, DataType.Str]:
        return x == y, "Value mismatch: {} vs {}".format(x, y)
    elif x_type == DataType.Float:
        return abs(x - y) < 1e-2, "Value mismatch: {} vs {}".format(x, y)
    elif x_type == DataType.TFTensor:
        return np.allclose(np.array(x), np.array(y), atol=1e-02, equal_nan=True), "Value mismatch: {} vs {}".format(x, y)
    elif x_type == DataType.KerasTensor:
        return np.allclose(np.array(x), np.array(y), atol=1e-02, equal_nan=True), "Value mismatch: {} vs {}".format(x, y)
    elif x_type in [DataType.List, DataType.Tuple]:
        if len(x) != len(y):
            return False, "Length mismatch: {} vs {}".format(len(x), len(y))
        for i in range(len(x)):
            equal, msg = is_equal(x[i], y[i])
            if not equal:
                return False, msg
        return True, None
    else:
        return False, "Unsupported type: {} <-- {}".format(x_type, type(x))

def check_code_randomness(code):
    if "tf.random" in code:
        return True
    if "dropout" in code.lower():
        return True
    return False

def check_less_possible_bug(code):
    if "tf.cast" in code:
        return True
    return False

def value_diff_type(code, msg):
    if check_code_randomness(code):
        return ResType.AllDiff_Rand
    elif check_less_possible_bug(code):
        return ResType.AllDiff_LessLikely
    elif "Type mismatch" in msg:
        return ResType.AllDiff_TypeMismatch
    return ResType.AllDiff

def is_allowed_err(error):
    error = str(error)
    allowed_error = [
        'tf.function only supports singleton tf.Variables created on the first call',
        'Using a symbolic `tf.Tensor` as a Python `bool` is not allowed',
        'len is not well defined for a symbolic Tensor',
        'To allow the shape to vary across iterations, use the `shape_invariants` argument of tf.while_loop to specify a less-specific shape.',
        'Python functions must return zero or more Tensors or ExtensionTypes or None values',
        'out of scope and cannot be used here',
        "This error may indicate that you're trying to pass a Tensor to a NumPy call, which is not supported.",
        "'SymbolicTensor' object has no attribute",
        "Iterating over a symbolic `tf.Tensor` is not allowed",
        "We failed to lift variable creations out of this tf.functio",
        "Attempting to capture an EagerTensor without building a function",
    ]
    for err in allowed_error:
        if err in error:
            return True
    return False

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def collect_trigger_info():
    trigger_cnt = Counter()
    logs = open(TFXLA_TRIGGER_LOG_PATH, 'r').readlines()
    for line in logs:
        optim_name = line.strip().rsplit(':', 1)[-1]
        if optim_name.endswith('.cc'):
            optim_name = optim_name[:-3]
        trigger_cnt[optim_name] += 1
    return trigger_cnt

def tfxla_test_executor_wrapper(exec_func, code, target_optim=None):
    # Empty the log file
    with open(TFXLA_TRIGGER_LOG_PATH, 'w') as f:
        f.write('')
    # Run the test
    ret, error = exec_func(code)
    # Collect the trigger info
    trigger_cnt = collect_trigger_info()
    if target_optim is not None:
        assert isinstance(target_optim, str)
        return ret, error, trigger_cnt[target_optim] > 0
    return ret, error, trigger_cnt

#TODO: handle multiple inputs with ast
def exec_wrapper(code):
    set_seed(RANDOM_SEED)
    exec(code, globals())
    model = m
    
    # This is a simple fix because the model likes to generate x instead of x1.
    # TODO: extract real input variable list from the code and avoid hard coded `x1`.
    # if 'm(x)' in code or 'x = ' in code:
    #     try:
    #         input_data = x
    #     except:
    #         pass
    # else:
    #     try:
    #         input_data = x1
    #     except Exception as e:
    #         raise e
    
    output = m(*input_data)
    #print(output)
    return output

def xla_run(code, mode="xla"):
    if mode == "xla":
        code = add_decorator(code, "@tf.function(jit_compile=True)")
        output = exec_wrapper(code)
        #print(tf.config.optimizer.get_jit())
    elif mode == "autocluster":
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        code = add_decorator(code, "@tf.function")
        #tf.config.optimizer.set_jit("autoclustering")
        #print(tf.config.optimizer.get_jit())
        output = exec_wrapper(code)
        os.environ['TF_XLA_FLAGS'] = ''
        #print(tf.config.optimizer.get_jit())
        #tf.config.optimizer.set_jit(False)
    elif mode == "naive":
        output = exec_wrapper(code)
    #print(output)
    return output



def nnsmith_run_tfxla_oracle(gen_path: Path, verbose=False):
    m, inputs = run_nnsmith.load_nnsmith_model(gen_path)
    model_fn = m.signatures["serving_default"]
    error = {
        "Naive Fail": None,
        "XLA Fail": None,
        "AC Fail": None,
        "Num Diff": None
    }

    fail = [0, 0, 0]
    allowed = [0, 0, 0]
    outputs = []
    #Test Failure & Num Inconsistency
    try:
        output = run_nnsmith.nnsmith_xla_run(model_fn, inputs, "naive")
        outputs.append(output)
    except Exception as e:
        if verbose:
            print("Naive Run failed.")
        error["Naive Fail"] = str(e)
        if is_allowed_err(e):
            allowed[0] = 1
        fail[0] = 1
    try:
        output = run_nnsmith.nnsmith_xla_run(model_fn, inputs, "xla")
        outputs.append(output)
    except Exception as e:
        #print(str(e))
        error["XLA Fail"] = str(e)
        if verbose:
            print("XLA Run failed.")
        if is_allowed_err(e):
            allowed[1] = 1
        fail[1] = 1
    try:
        output = run_nnsmith.nnsmith_xla_run(model_fn, inputs, "autocluster")
        outputs.append(output)
    except Exception as e:
        error["AC Fail"] = str(e)
        if verbose:
            print("Autocluster Run failed.")
        if is_allowed_err(e):
            allowed[2] = 1
        fail[2] = 1
    
    #Summarize Failure

    if fail != [0,0,0]:
        temp = []
        for i in range(3):
            temp += [fail[i] - allowed[i]]
        if temp == [0, 0, 0]:
            return ResType.AllPass, error
        return FAIL_MAPPING[str(fail)], error

    #Summarize Num Inconsistency
    if len(outputs) == 2: # We may not have this scenario
        equal, msg = is_equal(outputs[0], outputs[1])
        if not equal:
            error["Num Diff"] = "Yes"
            restype = ResType.AllDiff_TypeMismatch if "Type mismatch" in msg else ResType.AllDiff
            return restype, error
    elif len(outputs) == 3:
        equal, msg = is_equal(outputs[0], outputs[1])
        if not equal:
            error["Num Diff"] = msg
            restype = ResType.AllDiff_TypeMismatch if "Type mismatch" in msg else ResType.AllDiff
            return restype, error
        equal, msg = is_equal(outputs[0], outputs[2])
        if not equal:
            error["Num Diff"] = msg
            restype = ResType.AllDiff_TypeMismatch if "Type mismatch" in msg else ResType.AllDiff
            return restype, error
    
    return FAIL_MAPPING[str(fail)], error #ResType.AllPass

def run_tfxla_oracle(code, verbose=False):
    # Process the code at first.
    code = process_code(code)
    error = {
        "Naive Fail": None,
        "XLA Fail": None,
        "AC Fail": None,
        "Num Diff": None
    }

    fail = [0, 0, 0]
    allowed = [0, 0, 0]
    outputs = []
    #Test Failure & Num Inconsistency
    try:
        output = xla_run(code, "naive")
        outputs.append(output)
    except Exception as e:
        if verbose:
            print("Naive Run failed.")
        error["Naive Fail"] = str(e)
        if is_allowed_err(e):
            allowed[0] = 1
        fail[0] = 1
    try:
        output = xla_run(code, "xla")
        outputs.append(output)
    except Exception as e:
        #print(str(e))
        error["XLA Fail"] = str(e)
        if verbose:
            print("XLA Run failed.")
        if is_allowed_err(e):
            allowed[1] = 1
        fail[1] = 1
    try:
        output = xla_run(code, "autocluster")
        outputs.append(output)
    except Exception as e:
        error["AC Fail"] = str(e)
        if verbose:
            print("Autocluster Run failed.")
        if is_allowed_err(e):
            allowed[2] = 1
        fail[2] = 1
    
    #Summarize Failure

    if fail != [0,0,0]:
        temp = []
        for i in range(3):
            temp += [fail[i] - allowed[i]]
        if temp == [0, 0, 0]:
            return ResType.AllPass, error
        return FAIL_MAPPING[str(fail)], error

    #Summarize Num Inconsistency
    if len(outputs) == 2: # We may not have this scenario
        equal, msg = is_equal(outputs[0], outputs[1])
        if not equal:
            error["Num Diff"] = "Yes"
            restype = value_diff_type(code, msg)
            return restype, error
    elif len(outputs) == 3:
        equal, msg = is_equal(outputs[0], outputs[1])
        if not equal:
            error["Num Diff"] = msg
            restype = value_diff_type(code, msg)
            return restype, error
        equal, msg = is_equal(outputs[0], outputs[2])
        if not equal:
            error["Num Diff"] = msg
            restype = value_diff_type(code, msg)
            return restype, error
    
    return FAIL_MAPPING[str(fail)], error #ResType.AllPass

def test_Oracle():
    code = """
import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()


  def call(self, x1):
    x2 = tf.reshape(x1, [2,2])
    return tf.reshape(x2, [4])

m = Model()

input_shape = [4]
x1 = tf.constant([4.,5.,6.,7.], shape=input_shape)

y = m(x1)
    """
    ret, error, trigger_cnt = tfxla_test_executor_wrapper(run_tfxla_oracle, code)
    assert ret == ResType.AllPass
    assert 'ReshapeReshapeForwarding' in trigger_cnt
    assert 'IdentityReshapeRemoving' in trigger_cnt

    
    ret, error, trigger_cnt = tfxla_test_executor_wrapper(
        lambda c: (xla_run(c, "autocluster"), None), code)
    assert len(trigger_cnt) == 0

def test_Oracle2(): # you should see "debugs5"
    code = """
import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.w1 = tf.constant([1.,5.])


  def call(self, x1):
    x2 = tf.concat([x1, self.w1], 0)
    x3 = x2[2:4]
    return x3

m = Model()
input_shape = [2]
x1 = tf.constant([4.,5.], shape=input_shape)

y = m(x1)
    """
    ret, error, trigger_cnt = tfxla_test_executor_wrapper(run_tfxla_oracle, code)
    assert ret == ResType.AllPass
    assert 'SliceConcatForwarding' in trigger_cnt

    # More detailed analysis on each optimization mode
    
    ret, error, trigger_cnt = tfxla_test_executor_wrapper(
        lambda c: (xla_run(c, "naive"), None), code)
    assert len(trigger_cnt) == 0

    ret, error, trigger_cnt = tfxla_test_executor_wrapper(
        lambda c: (xla_run(c, "xla"), None), code)
    assert 'SliceConcatForwarding' in trigger_cnt
    assert 'ReshapeReshapeForwarding' in trigger_cnt
    assert 'IdentityReshapeRemoving' in trigger_cnt

    # Note: autocluster does not trigger `dynamic_dimension_simplifier.cc`
    # TODO: add test for autocluster
    ret, error, trigger_cnt = tfxla_test_executor_wrapper(
        lambda c: (xla_run(c, "autocluster"), None), code)
    assert len(trigger_cnt) == 0

def test_Oracle3(): # you should NOT see "debugs5"
    code = """
import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.w1 = tf.constant([[1.,5.],[6.,7.]])


  def call(self, x1):
    x2 = tf.concat([x1, self.w1], 0)
    x3 = x2[2:4]
    return x3

m = Model()
input_shape = [2,2]
x1 = tf.constant([[1., 2.], [4.,5.]], shape=input_shape)

y = m(x1)
    """
    ret, error, trigger_cnt = tfxla_test_executor_wrapper(run_tfxla_oracle, code)
    assert ret == ResType.AllPass
    assert 'IdentityReshapeRemoving' in trigger_cnt
    assert 'ReshapeReshapeForwarding' in trigger_cnt

    # More detailed analysis on each optimization mode
    
    ret, error, trigger_cnt = tfxla_test_executor_wrapper(
        lambda c: (xla_run(c, "naive"), None), code)
    assert len(trigger_cnt) == 0

    ret, error, trigger_cnt = tfxla_test_executor_wrapper(
        lambda c: (xla_run(c, "xla"), None), code)
    assert 'ReshapeReshapeForwarding' in trigger_cnt
    assert 'IdentityReshapeRemoving' in trigger_cnt

    # Note: autocluster does not trigger `dynamic_dimension_simplifier.cc`
    # TODO: add test for autocluster
    ret, error, trigger_cnt = tfxla_test_executor_wrapper(
        lambda c: (xla_run(c, "autocluster"), None), code)
    assert len(trigger_cnt) == 0


def test_random_input():
    code = """class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()

  def call(self, x1):
    a = (3 * x1,.6 * x1)
    x2 = tf.nn.relu(x1)
    a_3 = (x1,.6 - x1)   
    x3 = tf.pow(x1, 6.0)  
    return a[0] + a[1] + a_3[0] - x3

# Initializing the model
m = Model()

# Inputs to the model
input_shape = [10]
x1 = tf.random.normal(input_shape)

# Call model
y = m(x1)"""
    ret = run_tfxla_oracle(code)
    assert ret[0] == ResType.AllPass


def test_tree_rewrite():
    """Test the instrumentation after renaming works."""
    code = """
class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.conv = tf.keras.layers.Conv2D(96, 1, 1, "VALID")
    self.bn = tf.keras.layers.BatchNormalization()

  def call(self, x):
    x1 = tf.reduce_mean(self.bn(self.conv(x)), axis=[2, 3]) / 9.
    return x1 + tf.sigmoid(x1)

# Initializing the model
m = Model()

# Inputs to the model
input_shape = [1, 224, 224, 3]
x = tf.random.normal(input_shape)

# Call model
y = m(x)
"""
    ret, error, trigger_cnt = tfxla_test_executor_wrapper(run_tfxla_oracle, code)
    assert ret == ResType.AllPass
    # assert 'TreeReductionRewriter' in trigger_cnt


def test_null_comparison():
    code = """class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()

  def call(self, x1):
    x2 = tf.reshape(x1, [2,2])
    y = tf.reshape(x2, [4])

# Initializing the model
m = Model()

# Input to the model
input_shape = [2,2]
x1 = tf.reshape(tf.constant([4.,4.,5.,5.], shape=input_shape), [1,4,1,1],
)

# Call model
y = m(x1)"""
    ret = run_tfxla_oracle(code)
    assert ret[0] == ResType.AllPass

def test_tuple_comparison():
    code = """class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.m = tf.Variable([1., 2., 3., 4., 5., 6.])

  def call(self, x):
    return tf.math.top_k(self.m, k=2)


# Initializing the model
m = Model()

# Inputs to the model
x1 = tf.constant([1., 2., 3., 4., 5., 6.], shape=[1, 6])

print(m(x1))"""
    ret = run_tfxla_oracle(code)
    assert ret[0] == ResType.AllPass

def test_code_processing():
    code = """class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.m = tf.Variable([1., 2., 3., 4., 5., 6.])

  def __call__(self, x):
    return tf.math.top_k(self.m, k=2)


# Initializing the model
m = Model()

# Inputs to the model
x1 = tf.constant([1., 2., 3., 4., 5., 6.], shape=[1, 6])

print(m(x1))"""
    ret = run_tfxla_oracle(code)
    assert ret[0] == ResType.AllPass

def test_keras_tensor_comparison():
    code = """class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.fc1 = tf.keras.layers.Dense(2)
    self.fc2 = tf.keras.layers.Dense(2)
    self.fc3 = tf.keras.layers.Dense(2)

  def call(self, x):
    x = self.fc1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    return x
# Initializing the model
m = Model()

# Inputs to the model
x1 = tf.constant([[1.]], shape=[1,1,1])
x2 = tf.constant([[1., 2.]], shape=[1,2,1])
"""
    ret = run_tfxla_oracle(code)
    assert ret[0] == ResType.AllPass

def test_alldiff_lesslikely():
    code = """
class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()

    def all_reduce(self, x1):
        x2 = tf.reduce_sum(x1)
        x3 = tf.multiply(x2, 2.0)
        x4 = tf.multiply(x2, x2)
        return tf.add_n([x3, x4])

    def call(self, x1):
        out = (self.all_reduce(x1) + tf.math.reduce_prod(x1))
        return tf.cast(out, tf.uint8)
m = Model()
input_shape = [4]
x1 = tf.constant([4.0, 5.0, 6.0, 7.0], shape=input_shape)
input_data = [x1]
"""
    ret = run_tfxla_oracle(code)
    assert ret[0] == ResType.AllDiff_LessLikely

    
if __name__ == '__main__':
    test_Oracle()
    test_Oracle2()
    test_Oracle3()
    test_random_input()
    test_tree_rewrite()
    test_null_comparison()
    test_tuple_comparison()
    test_code_processing()
    test_keras_tensor_comparison()
    test_alldiff_lesslikely()
