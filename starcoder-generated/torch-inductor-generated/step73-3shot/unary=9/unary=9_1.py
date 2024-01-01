
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(8, 8, 9)
    def forward(self, x1):
        x2 = self.conv(3 + x1.clamp(min=0, max=6).div(6))
        x3 = 3 + x2.clamp(min=0, max=6).div(6)
        x4 = self.conv(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
