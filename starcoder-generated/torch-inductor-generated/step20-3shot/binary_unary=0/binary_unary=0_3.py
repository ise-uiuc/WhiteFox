
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1)
    def forward(self, x1, x2, x3, x4):
        t1 = self.conv1(x1)
        t2 = self.conv1(t1)
        t3 = t1 + x2
        t4 = torch.relu(t3)
        t5 = t2 + t4
        t6 = torch.relu(t5)
        t7 = self.conv2(t6)
        t8 = t7 + x3
        t9 = torch.relu(t8)
        t10 = t7 + x4
        t11 = torch.relu(t10)
        return t9, t11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
x4 = torch.randn(1, 16, 64, 64)
