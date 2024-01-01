
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv = torch.nn.Conv2d(3, 16, 2, stride=2, padding=5)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu6(v1)
        v3 = v1 + 3
        v4 = torch.clamp_min(v3, 0)
        v5 = torch.clamp_max(v4, 6)
        v6 = v1 * v5
        v7 = v6 / 6
        return v7
# Input to the model
x1 = torch.randn(2, 3, 28, 28)
