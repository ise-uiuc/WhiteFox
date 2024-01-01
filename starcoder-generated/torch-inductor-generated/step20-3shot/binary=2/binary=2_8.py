
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v3 = self.conv(x)
        v1 = v3 - 0.12
        v2 = v1 / 2.0
        return v2
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
