
class Layer1(torch.nn.Module):
    def __init__(self, inp, out, bias=True):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(inp, out, 3, 1, 1, bias=bias)
        self.conv2 = torch.nn.Conv2d(out, out, 3, 1, 1, bias=bias)
    def forward(self, v1):
        return self.conv1(v1) + self.conv2(v1)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Layer1(3, 16, 0)
        self.extra = Layer1(3, 16, 0)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
