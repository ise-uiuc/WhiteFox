
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(3)
        v3 = v2.clamp(max=6, min=0)
        out = v3.div(6)
        return out
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
