
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 16, 1, stride=1, padding=3)
    def forward(self, x):
        x = self.conv(x)
        y = x > 0.1
        z = x * -1.0
        a = torch.where(y, x, z)
        return a
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
