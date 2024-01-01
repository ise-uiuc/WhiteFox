
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(416, 196, 5, stride=1, padding=2, dilation=2)
        self.conv2d = torch.nn.Conv2d(196, 24, (1,1), stride=(1,1), bias=False, groups=1)
        self.batchnorm = torch.nn.BatchNorm2d(24, affine=True)
    def forward(self, x1, x2):
        v1 = self.conv_transpose1(x1)
        v2 = torch.cat((v1, x2), 1)
        v3 = self.conv2d(v2)
        v4 = self.batchnorm(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 416, 14, 14)
x2 = torch.randn(1, 24, 16, 16)
