
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(10, 5, 1, stride=1)
        self.conv3 = torch.nn.Conv2d(5, 3, 1, stride=1)
    def forward(self, V0):
        _v1 = self.conv1(V0)
        _v2 = _v1 - 0.463631
        _v3 = F.relu(_v2)
        _v4 = self.conv2(_v3)
        _v5 = _v4 - 2.963631
        _v6 = F.relu(_v5)
        _v7 = self.conv3(_v6)
        _v8 = _v7 - 3.1
        _v9 = F.relu(_v8)
        return _v9
# Inputs to the model
V0 = torch.randn(1, 3, 64, 64)
