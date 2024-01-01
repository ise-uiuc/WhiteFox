
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_before = torch.nn.Conv2d(2, 32, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
        self.conv_after = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dilation=1, groups=16)
    def forward(self, x2):
        v1 = self.conv_before(x2)
        v2 = self.conv_after(v1)
        v3 = -v2
        return v3
# Inputs to the model
x2 = torch.randn(1, 2, 32, 32)
