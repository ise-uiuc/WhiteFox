
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.relu6 = torch.nn.ReLU6()
        self.normalize = torch.nn.BatchNorm2d(num_features=8)
        self.normalize.weight.data.fill_(1e-5)
        self.normalize.bias.data.zero_()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(3)
        v3 = v2.clamp_min(0)
        v4 = v3.clamp_max(6)
        v5 = v4.div(6)
        return self.normalize(self.relu6(v5))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
