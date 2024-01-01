
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 1, 3)
        self.bn_first = torch.nn.BatchNorm2d(1)
        self.bn_second = torch.nn.BatchNorm2d(1, affine=False)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn_first(x)
        x = self.bn_second(x)
        return x
# Inputs to the model
x = torch.randn(2, 2, 5, 6)
