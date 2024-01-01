import json
import os
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Text, Any
import tensorflow as tf
from tensorflow import lite
from collections import Counter
from code_refiner import refine_code
import ast
import pdb

TFLITE_TRIGGER_LOG_PATH = "/tmp/tflite_trigger.log"

from enum import IntEnum, auto

class ResType(IntEnum):
    RANDOM = auto()
    STATUS = auto() # Means there's a potential bug: status inconsistency
    VALUE = auto() # Means there's a potential bug: value inconsistency
    PASS = auto()
    FAIL = auto()
    LITE_CONVERT_FAIL = auto()
    CRASH = auto()


def collect_trigger_info():
    trigger_cnt = Counter()
    logs = open(TFLITE_TRIGGER_LOG_PATH, 'r').readlines()
    for line in logs:
        optim_name = line.strip().rsplit(':', 1)[-1]
        trigger_cnt[optim_name] += 1
    return trigger_cnt
    
def tflite_test_executor_wrapper(exec_func, code, target_optim=None) -> Tuple[ResType, Dict[Text, Any], Dict[Text, Any]]:
    # Empty the log file
    with open(TFLITE_TRIGGER_LOG_PATH, 'w') as f:
        f.write('')
    # Run the test
    ret, error = exec_func(code)
    # Collect the trigger info
    trigger_cnt = collect_trigger_info()
    if target_optim is not None:
        assert isinstance(target_optim, str)
        return ret, trigger_cnt[target_optim] > 0
    return ret, error, trigger_cnt

def _evaluateTFLiteModel(tflite_model, input_data):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    for i in range(len(input_details)):
        # input_shape = input_details[i]['shape']
        # input_data_i = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[i]['index'], input_data[i])

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = [interpreter.get_tensor(output_details[i]['index'])
                   for i in range(len(output_details))]
    return output_data

def is_allowed_err(e: str) -> bool:
    allowed_errors = [
        "Failed to convert value into readable tensor",
        "Invalid tensor size",
        "Cannot set tensor: Dimension mismatch",
        "NumElements(bias)",
        "BytesRequired number of elements overflowed",
        "num_input_elements != num_output_elements",
        ":Cannot set tensor: Got value of type FLOAT64",
    ]
    for allowed_e in allowed_errors:
        if allowed_e in e:
            return True
    return False
        
def is_allowed_random(code: str) -> bool:
    allowed_randoms = [
        "random",
        "dropout"
    ]
    for allowed_r in allowed_randoms:
        if allowed_r in code.lower():
            return True
    return False

def run_tflite_oracle(code, verbose=True) -> Tuple[ResType, Dict[Text, Any]]:
    error = {
        "model_definition_error": None,
        "model_direct_inf_error": None,
        "model_convertion_error": None,
        "lite_model_inference_error": None
    }
    try:
        code_refined = refine_code(code)
        print(code_refined)
        exec(code_refined, globals())
        # TODO: change this to be the actual model name and input tensor names.
        model = m
        # input_data = input_data
    except Exception as e:
        # if verbose: print(e)
        error["model_definition_error"] = e.__class__.__name__ + ': ' + str(e)
        if is_allowed_err(str(e)):
            return ResType.FAIL, error
        return ResType.FAIL, error
    try:
        expected_value = model(*input_data)
        if type(expected_value) == tuple:
            list(expected_value)
        else:
            expected_value = [expected_value]
        expected_value_temp = []
        for v in expected_value:
            if type(v).__name__ == "TopKV2":
                expected_value_temp.append(v.values.numpy())
                expected_value_temp.append(v.indices.numpy())
            else:
                expected_value_temp.append(v.numpy())
        expected_value = sorted(expected_value_temp, key=lambda x: x.shape[0])
        expected_value = np.sort(expected_value, axis=None)
        
        #print(expected_value)
    except Exception as e:
        # if verbose: print(e)
        error["model_direct_inf_error"] = e.__class__.__name__ + ': ' + str(e)
        if is_allowed_err(str(e)):
            return ResType.FAIL, error
        return ResType.FAIL, error

    # Convert to tflite_model
    def test_func():
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        return tflite_model
    try:
        code_refined_tf = refine_code(code_refined, only_tf_func=True)
        exec(code_refined_tf, globals())
        # Previously model is defined above using the not refined code.
        # Now fix it by assigning model to the latest m
        model = m
        # input_data = input_data
        tflite_model = test_func()
    except Exception as e:
        print(str(e))
        error["model_convertion_error"] = e.__class__.__name__ + ': ' + str(e)
        return ResType.LITE_CONVERT_FAIL, error
    # Check crash
    try:
        actual_value = _evaluateTFLiteModel(tflite_model, input_data) 
        #pdb.set_trace()    
        actual_value_temp = []
        for v in actual_value:
            if type(v).__name__ == "TopKV2":
                actual_value_temp.append(v.values.numpy())
            else:
                actual_value_temp.append(v)
        actual_value = sorted(actual_value, key=lambda x: x.shape[0])
        actual_value = np.sort(actual_value, axis=None)
    except Exception as e:
        # if verbose: print(e)
        error["lite_model_inference_error"] = e.__class__.__name__ + ': ' + str(e)
        # Note: this is a catchabe exception, not a crash.
        return ResType.STATUS, error

    # Check inconsitent results
    try:
        tf.test.TestCase().assertAllClose(expected_value, actual_value, atol=1e-02)
    except:
        if is_allowed_random(code_refined_tf):
            return ResType.Pass, error
        if verbose:
            print(f"expected_value: {expected_value}")
            print(f"actual_value: {actual_value}")
        return ResType.VALUE, error
    
    return ResType.PASS, error

def testOracle():
    code = """
class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.w = tf.Variable([[1., 2.], [3., 4.]])
    self.m = tf.Variable([5., 6.])

  def call(self, x1):
    return tf.matmul(x1, self.w) * self.m

# Initializing the model
m = Model()

# Inputs to the model
input_shape = [1, 2]
x1 = tf.constant([7., 8.], shape=input_shape)

# Call model
y1 = m(x1)
"""
    ret, error, trigger_cnt = tflite_test_executor_wrapper(run_tflite_oracle, code)
    print(ret, error)
    assert ret == ResType.PASS
    assert 'FuseFullyConnectedAndMul' in trigger_cnt
    
def testOracleStaticShape():
    # The following code is taken from 
    # output/starcoder/tflite/srcnl2test-feedback-iter2-trigger/ConvertTrivialTransposeOpToReshapeOp/ConvertTrivialTransposeOpToReshapeOp_15.py
    # This is an example test to test optimizations requiring statis shape
    code = """class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.w1 = tf.Variable([1., 2.])

  def call(self, inp):
    return tf.transpose(inp, perm = [1, 0])

# Initializing the model
m = Model()

# Inputs to the model
input_shape = [2, 1]
x1 = tf.constant([[[1., 2.]]], shape=input_shape)
# Call Model
y1 = m(x1)"""
    ret, error, trigger_cnt = tflite_test_executor_wrapper(run_tflite_oracle, code)
    assert ret == ResType.PASS
    assert 'ConvertTrivialTransposeOpToReshapeOp' in trigger_cnt

def test_FuseFullyConnectedAndAdd():
    code = """
class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.w = tf.Variable([[3., 4.], [5., 6.]])
    self.b = tf.Variable([1., 2.])

  def call(self, x):
    return tf.matmul(x, self.w) + self.b

# Initializing the model
m = Model()

# Inputs to the model
x1 = tf.constant([1., 2.], shape=[1, 2])

"""
    ret, error, trigger_cnt = tflite_test_executor_wrapper(run_tflite_oracle, code)
    assert ret == ResType.PASS
    assert 'FuseFullyConnectedAndAdd' in trigger_cnt

def test_ConvertTrivialTransposeOpToReshapeOp():
    code = """class Model(tf.Module):

  def __init__(self):
    super(Model, self).__init__()
    self.w = tf.Variable([[3., 4.], [5., 6.]], trainable=True)

  @tf.function(input_signature=[tf.TensorSpec([1, 2], dtype=tf.float32)])
  def __call__(self, x):
    return tf.add(tf.linalg.matvec(self.w, x), self.w)

# Initializing the model
m = Model()

# Inputs to the model
x1 = tf.constant([1., 2.], shape=[1, 2])

"""
    ret, error, trigger_cnt = tflite_test_executor_wrapper(run_tflite_oracle, code)
    assert ret == ResType.PASS
    assert 'ConvertTrivialTransposeOpToReshapeOp' in trigger_cnt

def test_filter1():
    code = """
class Model(tf.keras.Model):

    def __init__(self):
        super(Model, self).__init__()
        self.w1 = tf.Variable([[3., 4.], [5., 6.]])
        self.b1 = tf.Variable([-4., 5.])
        self.c1 = tf.Variable([-7., 8.])
        self.w4 = tf.Variable([[13., 14.], [15., 16.]])
        self.b4 = tf.Variable([-14., 15.])

    def call(self, x1):
        x2 = tf.add(x1, self.c1)
        x3 = tf.matmul(x2, self.w1)
        x4 = tf.add(x3, self.b1)
        x5 = tf.add(x1, self.w4)
        x6 = tf.matmul(x5, self.w1)
        x7 = tf.add(x6, self.b4)
        x8 = tf.multiply(x4, x7)
        return tf.sigmoid(x4), tf.sigmoid(x8)

# Initializing the model
m = Model()

# Inputs to the model
input_shape = [1, 2]
x = tf.constant([1., 2.], shape=input_shape)

    """
    return run_tflite_oracle(code)

def test_filter2():
    code = """
class Model(tf.keras.Model):

  def call(self, x):
    return tf.math.top_k(x, 1)

# Initializing the model
input_shape = [5, 4]
m = Model()

# Input to the model
x = tf.constant([[1, 5, 2, 4], [3, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16],
                 [17, 18, 19, 20]], shape=input_shape)
x = x/2
    """
    return run_tflite_oracle(code)

def testMultiInputs():
    code = """
class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()

  def call(self, x1, x2):
    return x1 + x2 + x2

# Initializing the model
m = Model()

# Inputs to the model
input_shape = [1,]
x1 = tf.constant(7., shape=input_shape)

# Call model
y1 = m(x1)
"""
    ret, error, trigger_cnt = tflite_test_executor_wrapper(run_tflite_oracle, code)
    print(ret, error)
    assert ret == ResType.PASS

if __name__ == '__main__':
    # testOracle()
    # testOracleStaticShape()
    #test_FuseFullyConnectedAndAdd()
    #test_ConvertTrivialTransposeOpToReshapeOp()
    print(test_filter2())
    #print(test_filter1())
    testMultiInputs()
    