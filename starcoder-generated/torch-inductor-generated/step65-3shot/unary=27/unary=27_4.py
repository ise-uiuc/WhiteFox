
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 1, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.conv1(v3)
        v5 = torch.clamp_min(v4, self.min)
        v6 = torch.clamp_max(v5, self.max)
        v7 = self.conv2(v6)
        return v7
min = 1
max = 0
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
