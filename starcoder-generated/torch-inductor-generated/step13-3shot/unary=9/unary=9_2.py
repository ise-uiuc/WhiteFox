
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.add(3)
        v3 = torch.max(v2, 0)
        v4 = v3.min(6)
        v5 = v4.div(6)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
