
class Block(torch.nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.op1 = torch.nn.Sequential(torch.nn.BatchNorm2d(inp), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(inp, hidden, 1, 1, 0, bias=False))
        self.op2 = torch.nn.Sequential(torch.nn.BatchNorm2d(hidden), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(hidden, hidden, 1, 1, 0, bias=False))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        op1 = self.op1(concatenated_tensor)
        op2 = self.op2(op1 + concatenated_tensor)
        op3 = op1 + op2
        op4 = op3 + op1
        return torch.nn.ReLU()(op4 + v1)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Block(32, 16, 32)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
