
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        x2 = self.conv1(x1)
        t1 = self.conv2(x2)
        x4 = self.conv1(x1)
        t2 = self.conv2(x4)
        x6 = self.conv1(x1)
        t3 = self.conv2(x6)
        x7 = self.conv1(x1)
        t4 = self.conv2(x7)
        x9 = self.conv1(x1)
        t5 = self.conv2(x9)
        x8 = self.conv1(x1)
        t6 = self.conv2(x8)
        x10 = self.conv1(x1)
        t7 = self.conv2(x10)
        t1001 = torch.relu(t1 + t2 + t3 + t4 + t5 + t6 + t7)
        t1002 = self.conv1(x1)
        t1003 = torch.relu(t1001 + t1002)
        t1004 = torch.relu(t1003 + t1002)
        return t1004
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
