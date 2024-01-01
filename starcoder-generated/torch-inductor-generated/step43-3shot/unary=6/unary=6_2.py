
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        return torch.rsqrt(v1 + 11)
# Inputs to the model
x1 = torch.randn(1, 3, 31, 32)
