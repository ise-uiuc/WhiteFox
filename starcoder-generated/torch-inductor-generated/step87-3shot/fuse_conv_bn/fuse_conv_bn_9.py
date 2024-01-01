
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.bn = torch.nn.BatchNorm2d(3)
        self.pool2d = torch.nn.MaxPool2d(2)
        self.conv1 = torch.nn.Conv2d(3, 1, 2)
    def forward(self, x):
        x = self.conv(x)
        x = self.conv1(torch.exp(x))
        x = self.bn(x)
        x = self.pool2d(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
