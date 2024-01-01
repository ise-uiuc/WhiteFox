
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 2)
        self.bn = torch.nn.BatchNorm2d(3)
        self.activation = torch.nn.Sigmoid()
    def forward(self, x1):
        s = self.conv1(x1)
        t = self.conv2(s)
        t = self.bn(t)
        y = self.activation(t)
        return s
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6)
