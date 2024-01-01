
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=5, stride=2, padding=2, dilation=2, groups=3)
    def forward(self, x1, padding1):
        v1 = self.conv(x1 + padding1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 14, 12)
padding1 = 0.1
