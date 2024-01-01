import ast
import astunparse
from tflite_code_process import process_code

class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.shape_dclr = None
        self.var_dclr = None
        self.model_code = None

    def visit_ClassDef(self, node):
        node.name = "Model"
        self.model_code = astunparse.unparse(node)
    
    def visit_Assign(self, node):
        var_list = node.targets
        var = var_list[0]
        if len(var_list) == 1 and type(var) == ast.Name:
            if var.id.startswith("x") and self.var_dclr == None:
                var.id = "x"
                self.var_dclr = astunparse.unparse(node)
            elif "shape" in var.id:
                self.shape_dclr = astunparse.unparse(node)

def prepose_x(code):
    root = ast.parse(code)
    #print(ast.dump(root, indent=4))
    visitor = CodeVisitor()
    visitor.visit(root)
    model_dclr = f"m = Model()"
    if visitor.shape_dclr and visitor.var_dclr:
        code = "\n".join([visitor.shape_dclr, visitor.var_dclr, visitor.model_code, model_dclr])
        return code
    elif visitor.var_dclr:
        code = "\n".join([visitor.var_dclr, visitor.model_code, model_dclr])
        return code
    return None

#filter droupout/random

def refine_code(code, only_tf_func=False):
    signature = "[tf.TensorSpec(shape=x.shape, dtype=x.dtype) for x in input_data]"
    if only_tf_func:
        if "    def call" in code:
            code = code.replace("    def call", f"    @tf.function(input_signature={signature})\n    def call")
        elif "  def call" in code:
            code = code.replace("  def call", f"  @tf.function(input_signature={signature})\n  def call")
        return code
    code = process_code(code)
    if code:
        #print(code)
        prefix = "import tensorflow as tf\nimport numpy as np\n"
        code = prefix+code
    return code


def test_refiner():
    code = '''
class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.w = tf.Variable([[3., 4.], [5., 6.]])
        self.b = tf.Variable([1., 2.])

    def call(self, x, y):
        return tf.matmul(x, self.w) + self.b + y

# Initializing the model
m = Model()

# Inputs to the model
x1 = tf.constant([1., 2.], shape=[1, 2])
y = tf.constant([1., 2.], shape=[1, 2])
'''        
    code = refine_code(code)
    print(code)

if __name__ == "__main__":
    test_refiner()