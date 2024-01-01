
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
