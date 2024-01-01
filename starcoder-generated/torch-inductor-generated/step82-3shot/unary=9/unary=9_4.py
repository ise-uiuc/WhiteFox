
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.norm = torch.nn.GroupNorm(4, 8)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 3
        v3 = self.relu6(v2)
        v4 = v3 / 6
        v5 = self.norm(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
