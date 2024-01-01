
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.batchnorm = torch.nn.BatchNorm2d(8, affine=True)
    def forward(self, x1, other1=None, other2=None):
        v1 = self.conv(x1)
        if other1 == None:
            other1 = torch.randn(v1.shape)
        v2 = self.batchnorm(v1)
        v3 = v2 + other1
        if other2 == None:
            other2 = 0
        v4 = v3 + other2
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 55, 55)
