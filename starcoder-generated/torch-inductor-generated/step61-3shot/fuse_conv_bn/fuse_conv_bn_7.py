
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, bias=False)
        self.bn = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(10, 3, 1, bias=False)
        self.relu = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(6, 3, 1)
    def forward(self, x):
        x1 = self.conv1(x)
        x = self.relu(x1)
        x = self.bn(x1)
        x2 = self.conv2(x)
        v = torch.cat((x1, x2), dim=1)
        y = self.conv3(v)
        y1 = torch.add(y, 1)
        y2 = torch.add(y, 2)
        y3 = torch.add(y, 3)
        return y3
# Inputs to the model
x1 = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.int)
