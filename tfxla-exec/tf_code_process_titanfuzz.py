r"""
Usage:

TASK_DIR=../tfxla-exec/titanfuzz_example_outputs/
TASK_DIR=../output/titanfuzz/tf/
rm -r ../output/titanfuzz/trigger/
rm -r ../output/titanfuzz/log/
python run_tfxla.py --task_dir=${TASK_DIR} \
      --trigger_info_path=../output/titanfuzz/trigger/ \
      --res_dir=../output/titanfuzz/ --test_dir=../output/titanfuzz/log/ --titanfuzz \
 --trigger_info_path=../output/titanfuzz/trigger/trigger_info.jsonl
      

      
TASK_DIR=../output/titanfuzz/tf/
rm -r ../output/titanfuzz/trigger/
rm -r ../output/titanfuzz/log/
python tfxla_code_exec.py --task_dir=${TASK_DIR} \
      --trigger_info_path=../output/titanfuzz/trigger/ \
--res_dir=../output/titanfuzz/ --test_dir=../output/titanfuzz/log/ --test_log_path=../output/titanfuzz/log/tested.log \
 --temp_log_path=../output/titanfuzz/log/temp_code.py --titanfuzz \
 --trigger_info_path=../output/titanfuzz/trigger/trigger_info.jsonl


# To get optimization trigger information:
python eval.py --trigger_raw_json=../output/titanfuzz/trigger/trigger_info.jsonl
"""
import tensorflow as tf
import numpy as np
from pathlib import Path
import ast

def pass_syntax_check(code: str):
    try:
        ast.parse(code)
        return True
    except:
        return False

def syntax_pass_lines(code: str):
    """Remove last line until parse."""
    if pass_syntax_check(code):
        return code.splitlines()
    
    lines = code.splitlines()
    for idx in range(1, len(lines)):
        cur_lines = lines[:-idx]
        if pass_syntax_check('\n'.join(cur_lines)):
            return cur_lines
    return code.splitlines()

def isinit(line):
    if not "=" in line: return False
    var_name, init_stmt = line.split("=", 1)
    try:
        x = eval(init_stmt)
    except Exception as e:
        return False
    return True

class TFCodeGenerator():
    INDENT_SPACE = "    "
    IMPORTS = '\n'.join([
        'import os',
        'import tensorflow',
        'import tensorflow as tf',
        'import numpy as np',
        'import math',
    ])
    @staticmethod
    def indent(code, level=1):
        return '\n'.join([TFCodeGenerator.INDENT_SPACE * level + line
            for line in code.splitlines()])

    @staticmethod
    def generate(code):
        class_def_code, tensors, tensor_inits= TFCodeParser().split_func_tensor(code)
        imports = TFCodeGenerator.IMPORTS
        class_def_code = imports + '\n' + class_def_code + '\n' + tensor_inits + '\n'

        try:
            exec(class_def_code, locals(), locals())
            tensor_args = ', '.join(tensors)
            exec(f'm({tensor_args})', locals(), locals())
        except Exception as e:
            if 'dtype' in str(e):
                converts = '\n'.join([
                    f'{t} = tf.convert_to_tensor({t}, dtype=tf.float32)'
                    for t in tensors
                ])
                class_def_code += converts
        return class_def_code, tensors


class TFCodeParser():
    def split_func_tensor(self, code):
        code = self.process_code(code)
        lines = syntax_pass_lines(code)
        # Contains inits for tf.keras.layers.*
        layer_inits = ''
        # Contains the actualy function body
        func_body = ''
        # Contains the list of tensor argument names
        tensors: list[str] = []
        # Contains the initialization code
        tensor_inits = ''
        # Contains the list of return variables
        returns: list[str] = []
        imports = ''
        
        idx = 0
        total_line = len(lines)
        while idx < total_line:
            line = lines[idx]
            
            while not pass_syntax_check(line) and idx < total_line - 1:
                idx += 1
                line += lines[idx].strip()
            if idx >= total_line: break

            if isinit(line):
                # print('init code:', line)
                var_name, init_stmt = line.split("=", 1)
                var_name = var_name.strip()
                # print('var_name: ', var_name, var_name.isidentifier())
                if var_name.isidentifier():
                    if init_stmt.strip().startswith('tf.keras.layers.'):
                        layer_inits += 'self.' + line.strip() + '\n'
                        func_body = self.add_func_body(func_body, f'{var_name} = self.{var_name}')
                    else:
                        line = self.process_init_code(line)
                        tensors.append(var_name)
                        tensor_inits += line + '\n'
                else:
                    func_body = self.add_func_body(func_body, line)
            elif '=' in line:
                # print('func code:', line)
                if 'tf.keras.models.Model' in line:
                    idx += 1
                    continue
                var_name, assign_stmt = line.split('=', 1)
                var_name = var_name.strip()
                if var_name.isidentifier() and var_name not in returns:
                    returns.append(var_name)
                func_body = self.add_func_body(func_body, line)
            elif 'import' in line:
                imports += line + '\n'

            idx += 1

        tensor_args = ', '.join(tensors)
        return_args = ', '.join(returns)
        
        func_body = '\n'.join([
            'class MyModel(tf.keras.Model):',
            '',
            TFCodeGenerator.indent('def __init__(self):', 1),
            TFCodeGenerator.indent('super(MyModel, self).__init__()', 2),
            TFCodeGenerator.indent(layer_inits, 2),
            TFCodeGenerator.indent(f'def call(self, {tensor_args}):', 1),
            TFCodeGenerator.indent(func_body, 2),
            TFCodeGenerator.indent(f'return {return_args}', 2),
            '',
            'm = MyModel()'])
        return func_body, tensors, tensor_inits
        
    @staticmethod
    def process_func_body(line):
        return line

    @staticmethod
    def add_func_body(func_body, line):
        func_body += TFCodeParser.process_func_body(line) + '\n'
        return func_body

    @staticmethod
    def process_init_code(line: str):
        # TODO: handle inputs init with `tf.keras.Input`
        return line

    @staticmethod
    def process_code(code: str):
        lines = code.splitlines()
        new_lines = []
        for line in lines: 
            line = line.replace('\t', '    ')
            if 'assert' in line.lower():
                continue
            elif 'tf.print' in line.lower():
                continue
            else:
                new_lines.append(line)
        return '\n'.join(new_lines) + '\n'

if __name__ == '__main__':
    root_path = Path('titanfuzz_example_outputs/')
    # src = root_path / 'tf.abs_181.py'
    src = root_path / 'tf.keras.layers.ZeroPadding1D_176.py'
    class_def_code, tensors = TFCodeGenerator.generate(src.read_text())
    print(class_def_code)
    print('tensors: ', tensors)

    try:
        exec(class_def_code)
        tensor_args = ', '.join(tensors)
        exec(f'print(m({tensor_args}))')
    except Exception as e:
        print(e)