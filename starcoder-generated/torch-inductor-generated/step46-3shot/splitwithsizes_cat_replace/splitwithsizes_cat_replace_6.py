
class Layer1(torch.nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.op1 = torch.nn.Sequential(torch.nn.ReLU(True), torch.nn.Conv2d(inp, hidden, 1, 1, 0, bias=False))
        self.op2 = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Linear(hidden, out, bias=True))
    def forward(self, v1):
        return self.op2(self.op1(v1))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Layer1(32, 16, 64)
    def forward(self, x):
        split_tensors = torch.split(x, [1, 1, 1], dim=-1)
        concatenated_tensor = torch.cat(split_tensors, dim=-1)
        return (concatenated_tensor.reshape((concatenated_tensor.shape[0], -1)), torch.split(x, [1, 1, 1], dim=-1))
# Inputs to the model
x = torch.randn(2, 32, 10, 10)
