
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.conv = torch.nn.Conv2d(2, 1, 3, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x0):
        v1 = self.relu1(x0)
        v2 = self.conv(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = -0.3
max = 0.1
# Inputs to the model
x0 = torch.randn(1, 2, 10, 10)
