
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op1 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, 1, 0, bias=False))
        self.op2 = torch.nn.Sequential(torch.nn.BatchNorm2d(64))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (self.op1(concatenated_tensor), self.op2(concatenated_tensor))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Block()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return torch.nn.ReLU()(self.features(concatenated_tensor)[1])
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
