
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(2)
    def forward(self, x1):
        x = self.bn(x1)
        return x
# Inputs to the model
x1 = torch.randn(1, 2, 5, 5)
# model ends

# Generate a test script from a PyTorch model. The generated test script can be used with PyTorch and ORT to test whether the exported models from PyTorch meet the specfied requirements above. 
import numpy as np
np.random.seed(0)
torch.manual_seed(0)

from torchvision.models.alexnet import alexnet

m = alexnet(pretrained=False)
inputs = np.fromfile(
    'inputs.bin', dtype=np.float32).reshape(1, 3, 224, 224).astype(
        np.float32)
pytorch_out = m.eval()(torch.from_numpy(inputs))
m.eval().save("m.pt")
m.eval().onnx().save("m.onnx")
pytorch_out.detach().numpy().tofile('pytorch_out.bin')

import os
ortdir = os.path.dirname(os.path.abspath(__file__))+"/../../../../onnxruntime"
run(["python", ortdir+"/onnxruntime/test/python/bert_model_optimization.py", "m.onnx"])