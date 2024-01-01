
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.batchnorm = torch.nn.BatchNorm2d(3, affine=True)
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1)
    def forward(self, x):
        y = self.conv(F.relu(self.batchnorm(x)))
        return x * y
# Inputs to the model
x = torch.randn(1, 3, 128, 128)
