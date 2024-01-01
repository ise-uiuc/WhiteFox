import ast
import astunparse
import tensorflow as tf

def process_code(code: str) -> str:
    code = code.replace('__call__', 'call')
    parser = TFCodeParserJIT()
    class_code, class_name, tensors, tensor_inits = parser.split_func_tensor(code)
    code = class_code + "\n" + tensor_inits + "\n" + f"input_data = [{', '.join(tensors)}]\n"
    return code

class MultilineAssignTransformer(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Tuple):
            if len(node.targets[0].elts) == len(node.value.elts):
                return [ast.Assign(targets=[t], value=v) for t, v in zip(node.targets[0].elts, node.value.elts)]
        return node

class TFAssignRemover(ast.NodeTransformer):
    def visit_Assign(self, node):
        if any(self.is_tf_attribute(target) for target in node.targets):
            return ast.Pass()
        return self.generic_visit(node)

    def is_tf_attribute(self, node):
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == 'tf':
                return True
            return self.is_tf_attribute(node.value)
        return False

class TFCodeParserJIT():
    def __init__(self) -> None:
        pass

    def split_func_tensor(self, code):
        # get the code of model
        code = self.preprocessing(code)
        tree = ast.parse(code)
        class_init_args = []
        class_init_required_args = []
        class_init_code = ""
        class_code = ""
        class_name = ""
        class_forward_args = []
        class_forward_required_args = []
        tensors: List[str] = []
        tensor_inits = ''
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_code += astunparse.unparse(node) + "\n\n"
                class_name = node.name

                # get the arguments the initiation of this class
                try:
                    init_method = next(node for node in ast.walk(node) if isinstance(node, ast.FunctionDef) and node.name == "__init__")
                
                    class_init_args = [arg.arg for arg in init_method.args.args[1:]]
                    defaults = init_method.args.defaults
                    class_init_required_args = class_init_args[:len(class_init_args) - len(defaults)]
                except Exception as e:
                    pass

                try:
                    forward_method = next(node for node in ast.walk(node) if isinstance(node, ast.FunctionDef) and node.name == "call")
                    class_forward_args = [arg.arg for arg in forward_method.args.args[1:]]
                    defaults = forward_method.args.defaults
                    class_forward_required_args = class_forward_args[:len(class_forward_args) - len(defaults)]
                except Exception as e:
                    pass

            elif isinstance(node, ast.Assign):
                value = node.value
                if isinstance(value, ast.Call):
                    # first check whether is initialization of the class
                    if isinstance(value.func, ast.Name) and value.func.id == class_name:
                        # first split the tensor arguments and non-tensor arguments
                        if len(value.args) >= len(class_init_required_args) and len(value.args) <= len(class_init_args):
                            class_init_code = "m = " + astunparse.unparse(value) + "\n"
                        else:
                            class_init_code = ""
                        continue

                    func = value.func
                    args = value.args
                    
                    try:
                        tgt = node.targets[0].id
                    except Exception as e:
                        continue

                    init_code = astunparse.unparse(node)
                    if tgt not in tensors: 
                        # we need the arg code
                        for arg in ast.walk(value):
                            if isinstance(arg, ast.Name):
                                init_code = self.find_name_in_tree(tree, arg.id) + '\n' + init_code
                            elif isinstance(arg, ast.Starred):
                                if isinstance(arg.value, ast.Name):
                                    init_code = self.find_name_in_tree(tree, arg.value.id) + '\n' + init_code
                                
                        # test whether is tensor
                        try:
                            exec(init_code)
                            if isinstance(eval(tgt), tf.Tensor):
                                tensors.append(tgt)
                                tensor_inits += init_code + '\n'
                            elif tgt in class_forward_args:
                                tensors.append(tgt)
                                tensor_inits += init_code + '\n'
                        except Exception as e:
                            pass

        class_init_args_code = ""
        for arg_name in class_init_required_args:
            class_init_args_code += self.find_name_in_tree(tree, arg_name, use_default=True) + "\n"
        if class_init_code != "":
            class_init_code = class_init_args_code + class_init_code
        else:
            class_init_code = class_init_args_code
            class_init_code += f"\nm = {class_name}({', '.join(class_init_required_args)})\n"
        class_code += "\n" + class_init_code

        if len(tensors) < len(class_forward_args):
            diff = len(class_forward_args) - len(tensors)
            for arg_name in class_forward_required_args:
                if arg_name not in tensors:
                    tensors.append(arg_name)
                    tensor_inits += f"{arg_name} = tf.constant(1.0, shape=[1])\n"
                    diff -= 1
                    if diff == 0: break

        if len(tensors) > len(class_forward_args):
            tensors = tensors[:len(class_forward_args)]

        return class_code, class_name, tensors, tensor_inits
    
    @staticmethod
    def preprocessing(code: str):
        code = code.replace("\t", "    ")
        new_lines = []
        for line in code.splitlines():
            if line.strip().startswith("assert") or line.strip().startswith("import"):
                continue
            new_lines.append(line)
        code = "\n".join(new_lines)

        tree = ast.parse(code)
        transformer = MultilineAssignTransformer()
        new_tree = transformer.visit(tree)

        tf_assign_remover = TFAssignRemover()
        new_tree = tf_assign_remover.visit(new_tree)
        code = astunparse.unparse(new_tree)

        return code

    @staticmethod
    def find_name_in_tree(tree, arg_name, use_default=False):
        for _n in tree.body:
            if isinstance(_n, ast.Assign):
                for _t in _n.targets:
                    if isinstance(_t, ast.Name) and _t.id == arg_name:
                        return astunparse.unparse(_n)
        if arg_name == "batch_size":
            return f"{arg_name} = tf.constant(1.0, shape=[1])"

        if use_default:
            return f"{arg_name} = tf.constant(1.0, shape=[1])"
        else:
            return ""


def test_process_code():
    code = """class FFNMHAModelV1(tf.keras.Model):
  def __init__(self):
    super(FFNMHAModelV1, self).__init__()
    self.dense = tf.keras.layers.Dense(4, activation=tf.nn.gelu)
    self.attention = tf.keras.layers.MultiHeadAttention(4, 2)

  def call(self, q, v, k, mask):
    x = tf.concat([q, v], axis=-1)
    x = self.dense(x)
    att = self.attention(x, x, x, mask)
    x = q + att
    return x

x = tf.random.uniform([1, 4, 8])
kv = tf.random.uniform([1, 4, 8])
m = FFNMHAModelV1()

with tf.GradientTape() as tape:
  q = tf.Variable(tf.random.normal([1, 4, 4]), trainable=True)
  y = m(q, kv, kv, None)

tape.watch(q)
x_grad, kv_grad = tape.gradient(y, [q, kv])"""
    print(process_code(code))

if __name__ == "__main__":
    test_process_code()