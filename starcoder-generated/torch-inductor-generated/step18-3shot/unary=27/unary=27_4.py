
class Model(torch.nn.Module):
    def __init__(self, v1, v2):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 32, 1, stride=2, padding=1)
        self.a = v1
        self.b = v2
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.a)
        v3 = torch.clamp_max(v2, self.b)
        return v3
v1 = 0.6
v2 = 0.9
# Inputs to the model
x1 = torch.randn(1, 32, 100, 100)
