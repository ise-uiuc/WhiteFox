
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 48.9
        v3 = v2.repeat(4, 1, 4, 4)
        v4 = v3 - 7.3
        return v4
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
