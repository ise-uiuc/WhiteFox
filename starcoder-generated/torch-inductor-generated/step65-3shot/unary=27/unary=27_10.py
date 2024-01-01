
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(1, 2, 2, stride=1, padding=0, groups=2)
        self.conv1 = torch.nn.Conv2d(2, 1, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.conv1(v3)
        return v4
min = 0
max = 1
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
