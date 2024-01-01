
class Layer1(torch.nn.Module):
    def __init__(self, inp, hidden, out):
        super(Layer1, self).__init__()
        self.conv1 = torch.nn.Conv2d(inp, 32, 3, 1, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(32, affine=False, track_running_stats=False)
    def forward(self, v1):
        return torch.nn.ReLU()(self.bn1(self.conv1(v1)))
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.features = Layer1(3, 16, 32)
        self.extra = torch.nn.ReLU()
    def forward(self, v1):
        split_tensors = torch.split(v1, [1, 1, 1], dim=1)
        concatenated_tensor = torch.cat(split_tensors, dim=1)
        return (concatenated_tensor, torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
