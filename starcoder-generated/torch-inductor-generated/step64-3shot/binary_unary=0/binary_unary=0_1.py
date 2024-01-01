
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 64, 7, stride=1, padding=3)
        self.batchnorm = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU()

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 1
        v3 = self.batchnorm(v2)
        v4 = v3 + 2
        v5 = self.conv(v4)
        v6 = v5 + 3
        v7 = self.conv(v6)
        v8 = self.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64, requires_grad=True)
