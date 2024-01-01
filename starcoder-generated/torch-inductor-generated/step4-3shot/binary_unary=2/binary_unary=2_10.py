
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.zeros_like(v1)
        v2 = v2 - 0.5
        v3 = v1 + v2
        v4 = F.tanhshrink(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 8, 8)
