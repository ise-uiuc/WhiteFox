
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, bias=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        y = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
