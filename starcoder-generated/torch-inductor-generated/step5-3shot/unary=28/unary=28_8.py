
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        return v3

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(1, 16)

# Generating the model
torch.onnx.export(m, x1, "fp16.onnx", input_names=["input_tensor"], output_names=["output"], opset_version=12)
