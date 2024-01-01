
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, stride=2, dilation=3)
        bn = torch.nn.BatchNorm2d(1)
        relu = torch.nn.ReLU(inplace=True)
        self.conv = conv
        self.bn = bn
        self.relu = relu
    def forward(self, x1):
        x = self.conv(x1)
        x = self.bn(x)
        x = self.relu(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
