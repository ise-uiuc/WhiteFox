
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 512, (3, 3), stride=1, padding=(1, 1), dilation=1, groups=1)
        self.conv2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=(1, 1), dilation=1, groups=1)
        self.batchnorm = nn.BatchNorm1d(2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v1 + v2
        v4 = v3.view(10000, 512 * 4 * 4)
        v4 = v4.relu()
        v5 = v4 / self.batchnorm.bias
        return v5
# Inputs to the model
x1 = torch.randn(1, 256, 64, 64)
