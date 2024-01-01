
class Block2(torch.nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.op1 = torch.nn.Sequential(torch.nn.BatchNorm2d(inp), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(inp, hidden, (1, 1), 1, (0, 0), bias=(False)))
        self.op2 = torch.nn.Sequential(torch.nn.BatchNorm2d(inp), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(inp, hidden, (1, 1), 1, (0, 0), bias=(False)))
    def forward(self, v1):
        op1_list = torch.split(self.op1(v1), [1, 1, 1], dim=1)
        op1 = torch.cat(op1_list, dim=1)
        op2_list = torch.split(self.op2(v1), [1, 1, 1], dim=1)
        op2 = torch.cat(op2_list, dim=1)
        return torch.nn.ReLU()(op1 + op2)
class Block1(torch.nn.Module):
    def __init__(self, inp, hidden, out):
        super().__init__()
        self.layer1 = Block2(inp, hidden, out)
    def forward(self, v1):
        op1_list = torch.split(self.layer1(v1), [1, 1, 1], dim=1)
        op1 = torch.cat(op1_list, dim=1)
        op2_list = torch.split(self.layer1(v1), [1, 1, 1], dim=1)
        op2 = torch.cat(op2_list, dim=1)
        # Return
        return op1, op2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Block1(32, 16, 32)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
