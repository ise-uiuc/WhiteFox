
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1)
    def forward(self, x, x1):
        v1 = self.conv(x)
        v2 = v1 - x1
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
x1 = torch.randn(1, 3, 64, 64)
