
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.batchnorm = torch.nn.BatchNorm2d(num_features=8, eps=1e-05, momentum=0.1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.batchnorm(v1)
        v3 = v2 - 0.923
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
