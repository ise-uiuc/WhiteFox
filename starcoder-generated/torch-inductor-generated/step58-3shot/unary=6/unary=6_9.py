
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(5, 9, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + 100
        v4 = v3.clamp_min(0)
        v5 = v4.clamp_max(10)
        v6 = v2 * v5
        v7 = v6 / 10
        return v7
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
