
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 10, 1, stride=1, padding=1)
        self.batchnorm = torch.nn.BatchNorm2d(10)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.batchnorm(v1)
        v3 = self.conv(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 10, 128, 128)
