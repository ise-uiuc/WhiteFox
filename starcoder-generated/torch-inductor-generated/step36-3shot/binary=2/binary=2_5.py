
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=0, groups=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 42
        return v2
# Inputs to the model
x = torch.randn(1, 6, 64, 64)
