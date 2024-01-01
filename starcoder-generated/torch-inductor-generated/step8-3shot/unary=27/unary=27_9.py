
class Model(torch.nn.Module):
    def __init__(self, min=1, max=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 1, stride=3, padding=1)
        self.min = min
        self.max = max
    def forward(self, input):
        v1 = self.conv(input)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
# Inputs to the model
input = torch.randn(2, 2, 16, 22)
