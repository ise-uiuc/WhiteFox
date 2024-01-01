
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        for inp1 in inp:
            x = torch.matmul(inp1.permute(0, 2, 1), inp1.permute(0, 2, 1))
            y = torch.matmul(x, torch.matmul(x, x))
            z = torch.matmul(y, inp1.permute(0, 2, 1))
            a = torch.matmul(x, z)
            b = torch.matmul(a, x)
            c = torch.matmul(b, x)
            d = torch.matmul(b, c)
            out = c.permute(0, 2, 1).contiguous()
        return out
y = [(torch.randn(2,2,2), torch.randn(2,2,2), torch.randn(2,2,2))]
net = Model()
x = net(y)

import torch.onnx as onnx
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        out = nn.Hardtanh(inplace=True)(x)
        return out
model = Model()
# Generate inputs
x = torch.randn(2,2,2, requires_grad=True)
dummy_input = torch.onnx.utils.make_tensor_from_onnx_node(model, (x,))
# Export the model
torch.onnx.export(model,
                dummy_input,
                "debug_20210224_torchbmm.onnx",
                export_params=True,
                opset_version=10,
                do_constant_folding=False,
                verbose=False,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=None,
                example_output=None)
import onnx
from onnx import numpy_helper
from onnx import helper
m = onnx.load("debug_20210224_torchbmm.onnx")
graph = m.graph
#graph.input[0].type.tensor_type.shape.dim[2].dim_param = "None"
for i in graph.node:
    for j, inp in enumerate(i.input):
        if inp == "20":
            i.input[j] = ""
    print(i)