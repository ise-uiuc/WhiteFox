
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, stride=1, padding=4, dilation=4, groups=8)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.5
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
