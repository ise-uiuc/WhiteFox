
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.mean()
        v3 = v2.add(3)
        v4 = torch.div(v3, 6)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
