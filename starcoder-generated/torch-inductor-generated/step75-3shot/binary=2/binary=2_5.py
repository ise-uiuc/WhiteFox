
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 21, 1, 19)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.3673
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
