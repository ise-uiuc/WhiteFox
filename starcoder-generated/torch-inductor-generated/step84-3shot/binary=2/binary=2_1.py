
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=2, padding=10, dilation=2)
    def forward(self, x4):
        v1 = self.conv(x4)
        v2 = v1 - 1.0
        return v2
# Inputs to the model
x4 = torch.randn(1, 1, 4, 4)
