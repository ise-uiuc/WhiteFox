
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 46, 9, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(46)
        self.conv2 = torch.nn.Conv2d(46, 45, 15, stride=1, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(45)
    def forward(self, x2):
        v1 = self.conv1(x2)
        v2 = self.bn1(v1)
        v3 = self.conv2(v2)
        v4 = self.bn2(v3)
        v5 = v4 * 0.5
        v6 = v4 * v4
        v7 = v6 * v4
        v8 = v7 * 0.044715
        v9 = v4 + v8
        v10 = v9 * 0.7978845608028654
        v11 = torch.tanh(v10)
        v12 = v11 + 1
        v13 = v5 * v12
        return v13
# Inputs to the model
x2 = torch.randn(5, 1, 15, 18)
