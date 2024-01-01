


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0, dilation=1, groups=1)
    def forward(self, x1):
        x2 = torch.transpose(x1, 0, 1)
        v1 = self.conv(x2)
        v2 = v1 - 0.0
        return v2
# Inputs to the model
x1 = torch.randn(3, 1, 64, 64)
