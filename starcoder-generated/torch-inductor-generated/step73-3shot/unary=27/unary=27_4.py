
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1, 1, 5, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(1, 1, 5, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(1, 1, 5, stride=1, padding=2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = self.conv2(v2)
        v4 = torch.clamp_max(v3, self.max)
        v5 = self.conv3(v4)
        v6 = torch.clamp_max(v5, self.max)
        v7 = self.conv4(x1)
        return v7
min = 0.5
max = 0.9
# Inputs to the model
x1 = torch.randn(1, 1, 100, 100)
