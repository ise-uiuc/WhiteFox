
class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 1, 1)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        return torch.cat([torch.sum(v, dim=[-1, -2, -3]) for v in split_tensors])
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential([Block()] * 50)
        self.extra = torch.nn.Softmax(dim=1)
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        return torch.cat([torch.sum(v, dim=[-1, -2, -3]) for v in split_tensors])
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
