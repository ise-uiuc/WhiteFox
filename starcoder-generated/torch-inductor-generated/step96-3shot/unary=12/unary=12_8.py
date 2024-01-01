
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=2, padding=0, groups=1)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=0, dilation=1)
        self.conv1 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid = torch.nn.Hardsigmoid()
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=3, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x1):
        v0 = self.conv(x1)
        v1 = self.maxpool(v0)
        v2 = self.conv1(v1)
        v3 = self.hardsigmoid(v2)
        v4 = self.conv2(v3)
        v5 = v2 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
