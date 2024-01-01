
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 3, stride=2, padding=6)
        self.min = min
        self.max = max
    def forward(self, input, min, max):
        v1 = self.conv(input)
        v2 = torch.clamp_min(v1, min)
        v3 = torch.clamp_max(v2, max)
        return v3
min = 0.8
max = 0.3
# Inputs to the model
input = torch.randn(1, 3, 52, 52)
min = -0.6
max = 0.9
