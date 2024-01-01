
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(2, 2, 2)
        self.batchnorm2d = torch.nn.BatchNorm2d(2)
    def forward(self, x):
        conv2d = self.conv2d(x)
        batchnorm2d = self.batchnorm2d(conv2d)
        return batchnorm2d
# Inputs to the model
x = torch.randn(1, 2, 4, 4)
