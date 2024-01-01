
class Layer1(torch.nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.op1 = torch.nn.Sequential(torch.nn.MaxPool2d(3, 1, 1), torch.nn.BatchNorm2d(3), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(inp, hidden, 1, 1, 0, bias=False), torch.nn.BatchNorm2d(hidden), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(hidden, out, 1, 1, 0, bias=False))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return self.op1(concatenated_tensor)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Layer1(32, 16, 32)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
