
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 13, stride=1, padding=3, dilation=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 5.5
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
