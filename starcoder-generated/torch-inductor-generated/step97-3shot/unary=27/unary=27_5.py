
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 579, 16, stride=4, padding=12)
        self.tanh = torch.nn.Tanh()
        self.max = max
        self.min = min
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.tanh(v1)
        v3 = torch.clamp_min(v2, self.min)
        v4 = torch.clamp_max(v3, self.max)
        return v4
min = 0.7
max = 22.4
# Inputs to the model
x1 = torch.randn(1, 1, 123, 211)
