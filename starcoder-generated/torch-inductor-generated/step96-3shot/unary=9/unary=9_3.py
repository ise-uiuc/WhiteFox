
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v2 = self.conv(x1)
        v3 = (v2).clamp(0, 6)
        v3 = v3 + 3
        return v3.div(6)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
