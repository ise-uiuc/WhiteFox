
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 4)
        self.batchnorm2d = torch.nn.BatchNorm2d(3)
    def forward(self, x1):
        s = self.conv1(x1)
        t = self.batchnorm2d(s)
        return t
# Inputs to the model
x1 = torch.rand(1, 3, 6, 6)
