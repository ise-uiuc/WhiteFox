
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 5, 2, stride=1, padding=1)
        self.min = min
        self.max = max
    def forward(self, input):
        v1 = self.conv(input)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = -0.5
max = 0.3
# Inputs to the model
input = torch.randn(1, 2, 64, 64)
