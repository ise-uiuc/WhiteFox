
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 4, 3, stride=2, padding=1)
    def forward(self, x):
        v3 = self.conv(x)
        v4 = 1.0 - v3
        v1 = v4 / 3.0
        return v1
# Inputs to the model
x = torch.randn(1, 2, 64, 64)
