
class Block(torch.nn.Module):
    def __init__(self, inp, middle, out):
        super().__init__()
        self.op1 = torch.nn.Sequential(torch.nn.BatchNorm2d(inp), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(inp, 128, 1, 1, 0, bias=False))
        self.op2 = Block(128, 128, 128)
        self.op3 = Block(128, 128, 128)
        self.op4 = Block(128, 128, 128)
        self.op5 = torch.nn.Sequential(torch.nn.BatchNorm2d(128), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(128, out, 1, 1, 0, bias=False))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        op1 = self.op1(concatenated_tensor)
        op2 = self.op2(op1 + concatenated_tensor)
        op3 = self.op3(op2 + op1)
        op4 = self.op4(op3 + op2)
        op5 = self.op5(op4 + op3)
        return torch.nn.ReLU()(op5 + v1)
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
