 with two batch norm ops
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.bn_1 = torch.nn.BatchNorm2d(10)
        self.bn_2 = torch.nn.BatchNorm2d(10)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.bn_1(t1)
        t3 = self.relu(t2)
        t4 = 2 * t3
        t5 = self.relu(t4)
        t6 = self.bn_2(t5)
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
