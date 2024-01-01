
class Model(torch.nn.Module):
    def __init__(self, a=1.0, b=0.0):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=3, padding=5)
        self.a = a
        self.b = b
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.a)
        v3 = torch.clamp_max(v2, self.b)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
