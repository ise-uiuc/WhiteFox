
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = torch.clamp_min(v1, self.min)
        v1 = torch.clamp_max(v1, self.max)
        v2 = self.relu(v1)
        return v2
min = 0.001
max = 0.7
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
