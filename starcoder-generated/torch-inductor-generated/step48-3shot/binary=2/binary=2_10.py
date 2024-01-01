
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), groups=1)
        self.batchnorm0 = torch.nn.BatchNorm2d(32)
    def forward(self, x2):
        v1 = self.conv0(x2)
        v2 = self.batchnorm0(v1)
        return v2 + -0.114
# Inputs to the model
x2 = torch.randn(1, 32, 24, 24)
