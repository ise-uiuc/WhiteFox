
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.mul = torch.nn.functional.mul
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, 6)
        v3 = torch.clamp_max(v2, 0)
        return self.mul(v1, v3)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
