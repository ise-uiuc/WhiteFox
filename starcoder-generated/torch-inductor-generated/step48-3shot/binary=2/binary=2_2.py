
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0, dilation=1, groups=1)
    def forward(self, x2):
        v1 = self.conv(x2)
        return -x2
# Inputs to the model
x2 = torch.randn(1, 32, 35, 35)
