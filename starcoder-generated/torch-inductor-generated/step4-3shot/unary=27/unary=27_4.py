
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1, stride=1, padding=0)
        self.min = min
        self.max = max
    def forward(self, input):
        v1 = self.conv(input)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.9
max = -0.4
# Inputs to the model
input = torch.randn(1, 1, 64, 64)
