
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 5, stride=1, padding=3)
        self.min = min
        self.max = max
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 0.4622644447597122
max = 0.5410665342330933
# Inputs to the model
x1 = torch.randn(9, 2, 22, 26)
