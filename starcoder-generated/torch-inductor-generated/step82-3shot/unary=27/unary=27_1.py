
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv = torch.nn.Conv2d(3, 4, 2, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.relu6(x1)
        v2 = self.conv(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0.0
max = 1.0
# Inputs to the model
x1 = torch.randn(1, 3, 40, 40)
