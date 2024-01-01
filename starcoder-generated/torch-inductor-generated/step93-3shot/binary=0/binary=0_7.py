
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 64, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1 / v1
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
