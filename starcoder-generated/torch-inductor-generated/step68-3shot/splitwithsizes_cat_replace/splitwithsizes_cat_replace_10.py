
class Block(torch.nn.Module):
    def __init__(self, op1):
        super().__init__()
        self.op1 = op1
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        op1 = self.op1(concatenated_tensor)
        op2 = op1 + concatenated_tensor
        op3 = op1 + op2
        op4 = op3 + op1
        return op4 + v1
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Block(torch.nn.Sequential(torch.nn.BatchNorm2d(32), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(32, 64, 1, 1, 0, bias=False)))
        self.features2 = Block(torch.nn.Sequential(torch.nn.BatchNorm2d(64), torch.nn.ReLU(inplace=False), torch.nn.Conv2d(64, 64, 1, 1, 0, bias=False)))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        op1 = self.features(concatenated_tensor)
        op2 = self.features2(op1 + concatenated_tensor)
        op3 = op1 + op2
        op4 = op3 + op1
        return op4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
