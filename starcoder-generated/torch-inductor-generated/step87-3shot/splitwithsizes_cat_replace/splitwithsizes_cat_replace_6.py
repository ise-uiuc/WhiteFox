
class Block(torch.nn.Module):
    def __init__(self, inp, hidden):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(inp, hidden, 3, 1, 1)
        self.bn1 = torch.nn.BatchNorm2d(hidden, affine=False, track_running_stats=False)
        self.conv2 = torch.nn.Conv2d(hidden, hidden, 3, 1, 1)
        self.bn2 = torch.nn.BatchNorm2d(hidden, affine=False, track_running_stats=False)
    def forward(self, x1):
        out = self.conv1(torch.nn.ReLU()(self.bn1(x1)))
        split_tensors = torch.split(out, [1, 1, 1], dim=1)
        return torch.cat(split_tensors, dim=1) + torch.nn.ReLU()(split_tensors[0] - split_tensors[1]) + torch.nn.Linear(hidden, 1, bias=False).cuda()(out)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Block(3, 16)
        self.extra = torch.nn.ReLU()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        return torch.cat(split_tensors, dim=1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
