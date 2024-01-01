
class Model(torch.nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 11, bias=False)
        self.batchnorm = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.batchnorm(v1)
        v3 = torch.relu(v2)
        v4 = self.pool(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
