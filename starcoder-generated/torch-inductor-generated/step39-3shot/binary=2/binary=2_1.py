
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_before = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)
        self.conv_after = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, groups=1)
    def forward(self, x1):
        v1 = self.conv_before(x1)
        v2 = v1 - 1669
        v3 = self.conv_after(v2)
        return -v3
# Inputs to the model
x1 = torch.randn(1, 32, 36, 36)
