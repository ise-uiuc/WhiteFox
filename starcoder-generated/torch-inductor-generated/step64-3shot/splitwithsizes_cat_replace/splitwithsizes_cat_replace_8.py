
class Block(torch.nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.op = torch.nn.Sequential(torch.nn.BatchNorm2d(inp), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(inp, hidden, 1, 1, 0, bias=False))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        op1 = self.op(concatenated_tensor)
        return torch.nn.ReLU()(op1 + v1)
class Layer1(torch.nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.features = Block(inp, hidden, out)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        op1 = self.features(concatenated_tensor)
        op2 = op1 + concatenated_tensor
        return torch.nn.ReLU()(op2 + v1)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Layer1(3, 16, 32)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
