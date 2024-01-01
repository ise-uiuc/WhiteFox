
class Block(torch.nn.Module):
    def __init__(self, inp, hidden, out):
        super(Block, self).__init__()
        self.conv1 = torch.nn.Conv2d(inp, 32, 3, 1, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(inp, 64, 3, 1, 0, bias=False)
        self.conv3 = torch.nn.Conv2d(64, out, 3, 1, 0, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32, affine=False, track_running_stats=False)
        self.bn2 = torch.nn.BatchNorm2d(64, affine=False, track_running_stats=False)
    def forward(self, x1):
        split_tensors = torch.split(x1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return torch.nn.ReLU()(self.bn1(self.conv1(x1)) + self.bn2(self.conv2(concatenated_tensor)) + self.conv3(concatenated_tensor))
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = torch.nn.Sequential(Block(3, 16, 32))
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
