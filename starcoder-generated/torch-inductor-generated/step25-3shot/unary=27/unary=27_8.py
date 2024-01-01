
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 8, 2, stride=1, padding=2)
        self.min = min
        self.max = max
    def forward(self, input, min, max):
        v1 = self.conv(input)
        v2 = torch.clamp_min(v1, min)
        v3 = torch.clamp_max(v2, max)
        return torch.flatten(v3, 1)
min = -9
max = 89
# Inputs to the model
input = torch.randn(1, 6, 52, 52)
min = -9
max = 89
