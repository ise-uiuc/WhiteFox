
class Layer1(torch.nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.op1 = torch.nn.Sequential(torch.nn.BatchNorm2d(inp), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(inp, hidden, 1, 2, 0, bias=False))
        self.op2 = torch.nn.Conv2d(hidden, out, 1, 2, 0, bias=False)
    def forward(self, v1):
        op1_res = self.op2(self.op1(v1))
        return op1_res
class Layer2(torch.nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.op1 = torch.nn.Sequential(torch.nn.Conv2d(inp, hidden, 1, 1, 0, bias=False), torch.nn.BatchNorm2d(hidden), torch.nn.ReLU(inplace=False))
        self.op2 = torch.nn.Sequential(torch.nn.Conv2d(hidden, hidden, 1, 1, 0, bias=False), torch.nn.BatchNorm2d(hidden), torch.nn.ReLU(inplace=False))
        self.op3 = torch.nn.Conv2d(hidden, out, 1, 1, 0, bias=False)
    def forward(self, v1):
        return self.op2(self.op1(v1) + v1) + self.op3(v1)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = [Layer1(32, 16, 16), Layer2(16, 8, 16)]
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
