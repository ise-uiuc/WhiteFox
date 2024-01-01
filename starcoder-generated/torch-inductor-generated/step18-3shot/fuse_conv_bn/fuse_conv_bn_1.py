
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)   # [1, 3, 6, 6]
        self.conv2 = torch.nn.Conv2d(3, 3, 2)   # [1, 3, 5, 5]
        self.bn = torch.nn.BatchNorm2d(3)
        self.activation = torch.nn.Sigmoid()
    def forward(self, x1):
        s = self.conv1(x1)                   # [1, 3, 6, 6]
        t = self.conv2(s)                    # [1, 3, 5, 5]
        t = self.bn(t)                       # [1, 3, 5, 5]
        y = self.activation(t)               # [1, 3, 5, 5]
        return s
# Inputs to the model
x1 = torch.randn(1, 3, 6, 6)
