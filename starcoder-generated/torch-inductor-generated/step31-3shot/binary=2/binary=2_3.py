
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        s = 1.0
        v2 = (v1 - s) * 12.0
        return v2
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
