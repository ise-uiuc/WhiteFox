
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, stride=3, padding=7, dilation=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.9
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
