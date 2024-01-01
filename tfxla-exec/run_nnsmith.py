"""

TASK_DIR=/home/src/JIT-parser/output/nnsmith/tensorflow-output/
RES_DIR=/home/src/JIT-parser/output/nnsmith/
rm -r /home/src/JIT-parser//output/nnsmith/trigger/
rm -r /home/src/JIT-parser//output/nnsmith/log/
python run_tfxla.py --task_dir=${TASK_DIR} \
      --trigger_info_path=${RES_DIR}/trigger/ \
      --res_dir=${RES_DIR}/ --test_dir=${RES_DIR}/log/ --nnsmith \
 --trigger_info_path=${RES_DIR}/trigger/trigger_info.jsonl

 
# To get optimization trigger information:
python eval.py --trigger_raw_json=../output/nnsmith/trigger/trigger_info.jsonl
 """
from pathlib import Path
import tensorflow as tf
import numpy as np
import random
import pickle
import os
import tempfile


RANDOM_SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def _evaluateTFLiteModel(tflite_model, input_data):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i in range(len(input_data)):
        interpreter.set_tensor(input_details[i]['index'], input_data[i])
    interpreter.invoke()
    output_data = [interpreter.get_tensor(output_details[i]['index'])
                   for i in range(len(output_details))]
    return output_data

def read_all_nnsmith_tasks(gen_root: Path):
    tasks = []

    for gen_path in gen_root.iterdir():
        if not gen_path.is_dir():
            continue
        
        label = str(gen_path)
        code = ''
        opt = ''
        tasks.append([opt, label, code])

    tasks = sorted(tasks, key=lambda x: x[1])
    return tasks

def load_nnsmith_model(generated_model_path: Path):
    """Returns a loaded TF model and its inputs"""
    model_path = generated_model_path / 'model' / 'tfnet'
    m = tf.saved_model.load(model_path)
    with open(generated_model_path / 'oracle.pkl', 'rb') as f:
        oracle = pickle.load(f)
    return m, oracle['input']



def nnsmith_xla_run(f, inputs, mode="xla"):
    if mode == "xla":
            
        cwd = os.getcwd()
        # Create a temp dir to execute the code
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)

            # Enter the temp dir to execute.
            os.chdir(tmpdirname)
            with tf.device('cpu'):
                compiled = tf.function(jit_compile=True)(f)
                output = compiled(**inputs)
                output = list(output.values())
                
            # Return to the previous dir
            os.chdir(cwd)

        #print(tf.config.optimizer.get_jit())
    elif mode == "autocluster":
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
        
        cwd = os.getcwd()
        # Create a temp dir to execute the code
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)

            # Enter the temp dir to execute.
            os.chdir(tmpdirname)
            with tf.device('cpu'):
                compiled = tf.function()(f)
                output = compiled(**inputs)
                output = list(output.values())
                
            # Return to the previous dir
            os.chdir(cwd)

        os.environ['TF_XLA_FLAGS'] = ''
    elif mode == "naive":
        cwd = os.getcwd()
        # Create a temp dir to execute the code
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirname = Path(tmpdirname)

            # Enter the temp dir to execute.
            os.chdir(tmpdirname)
            with tf.device('cpu'):
                output = f(**inputs)
                output = list(output.values())
                
            # Return to the previous dir
            os.chdir(cwd)
    #print(output)
    return output

if __name__ == '__main__':
    gen_root = Path('../output/nnsmith/tensorflow-output/')
    gen_path = gen_root / '423.163'
    model_path = gen_path / 'model' / 'tfnet'
    m = tf.saved_model.load(model_path)
    with open(gen_path / 'oracle.pkl', 'rb') as f:
        oracle = pickle.load(f)
    print(oracle['input'].keys())
    # x = oracle['input']['v3_0']
    print(m(**oracle['input']))
    # print(oracle)
    device_info = oracle['provider'] # 'tf[cpu] eager'
    exit(0)
    providers = set()
    model_cnt = 0
    for gen_path in gen_root.iterdir():
        try:
            with open(gen_path / 'oracle.pkl', 'rb') as f:
                oracle = pickle.load(f)
            providers.add(oracle['provider'])
            model_cnt += 1
        except:
            pass   
    print(providers)
    print(model_cnt)
    
    exit(0)
    
    # converter = tf.lite.TFLiteConverter.from_saved_model(str(model_path))
    # converter.target_spec.supported_ops = [ tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS ]
    # tflite_model = converter.convert()
    # actual_value = _evaluateTFLiteModel(tflite_model,[x])
    # print('tflite model output:')
    # print(actual_value)
    f = m.signatures["serving_default"]
    # print(f)
    # print(f(v3_0=x))
    with tf.device('cpu'):
        compiled = tf.function(jit_compile=True)(f)
        # print(compiled)
        print(compiled(v3_0=x))


