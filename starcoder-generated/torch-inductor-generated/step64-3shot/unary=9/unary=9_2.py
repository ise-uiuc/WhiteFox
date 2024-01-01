
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.add(torch.clamp(v1, min=0, max=6), 3)
        out = torch.div(v2, 6)
        return out
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
