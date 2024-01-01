
class Model(torch.nn.Module):
    def __init__(self, input_channels, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(input_channels, 100, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
input_channels = 100
min = 0.11954774620056152
max = 2.0198440551757812
# Inputs to the model
x1 = torch.randn(1, input_channels, 64, 64)
