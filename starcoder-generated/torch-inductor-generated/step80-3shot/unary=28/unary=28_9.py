
import io
import torch
import torch.nn
import torch.onnx

# Shape inference for aten::max_pool2d
def symbolic_max_pool2d(g, input, kernel_size, stride=None, padding=None, dilation=None, ceil_mode=False):
    return g.op("MaxPool", input, kernel_shape_i=kernel_size, pads_i=padding, strides_i=stride)

class OnnxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = torch.nn.Linear(64, 32)
        self.relu = torch.nn.ReLU()

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x1 = self.dense(x0)
        x2 = self.relu(x1)
        x3 = torch.ops.aten._max_pool2d('', x2)
        # The operator output is expected to be a tensor of rank 0.
        # So, the ONNX models always returns a tensor of rank 1 regardless whether the dimension size is equal to 0.
        # To workaround this issue, the output tensor of rank 1 is removed, returning just a tensor of arbitrary rank.
        shape = list(x3.shape)
        if shape[0] == 0:
            shape.remove(0)
        x4 = torch.reshape(x3, shape)
        return x4

    def from_session(self, sess):
        model = io.BytesIO()
        sess.serialize(model)
        torch.onnx._optimize_model(io.BytesIO(model.read()))

def onnx_model_factory(m):
    # A PyTorch model m that accepts a tensor of rank 5 where the input tensor has shape [batches x input channels x input depth x width x height]
    # and the output tensor has arbitrary rank whose total size is [batches x output channels x output depth x width x height].
    return OnnxModel()

onnx_model = torch.nn.Module()
x0 = torch.randn(1, 3, 27, 56)
onnx_model.eval()
onnx_model = torch.onnx.export(onnx_model, x0, "public_pt_onnx_model.onnx", opset_version=13, example_outputs=onnx_model(x0), input_names=["input"], output_names=["output"], custom_opsets={"onnx": 13})

