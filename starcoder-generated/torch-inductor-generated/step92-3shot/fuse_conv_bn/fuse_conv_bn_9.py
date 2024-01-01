
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3,3,3)
        self.linear = torch.nn.Linear(4,5)
        self.bn = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        h1 = self.conv(x).relu_()
        l1 = self.bn(x).relu_()
        o1 = self.linear(x)
        return o1
# Inputs to the model
x = torch.randn(10, 3, 4, 4)
