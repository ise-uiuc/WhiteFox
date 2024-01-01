
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        t1 = self.conv1(x1)
        t2 = self.conv2(x2)
        t3 = self.conv3(x3)
        t4 = t1 + t2
        t5 = torch.relu(t4)
        t6 = t3 + t5
        t7 = torch.relu(t6)
        t8 = t7 + t4
        t9 = torch.relu(t8)
        t10 = self.conv4(t9)
        t11 = t10 + t7
        t12 = torch.relu(t11)
        return t12
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
