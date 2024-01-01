
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, 5, stride=5, padding=2)
        self.conv2 = torch.nn.Conv2d(8, 8, 5, stride=5, padding=2)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0.1
max = 0.2
# Inputs to the model
x1 = torch.randn(1, 4, 100, 100)
