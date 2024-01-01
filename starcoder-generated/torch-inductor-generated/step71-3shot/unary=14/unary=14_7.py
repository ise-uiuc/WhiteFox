
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 30, 2, stride=1, padding=1, dilation=1, groups=3, bias=True)
        self.relu = torch.nn.ReLU6(inplace=True)
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(30, 8, 4, stride=1, padding=3, dilation=3, groups=2)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.relu(v1)
        v3 = self.conv_transpose_1(v2)
        return v3
x1 = torch.randn(1, 3, 88, 88)
