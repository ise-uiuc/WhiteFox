
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        v4 = self.conv2(v3)
        v5 = torch.clamp_min(v4, self.min)
        v6 = torch.clamp_max(v5, self.max)
        return v6
min = 2.37
max = 6.14
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
