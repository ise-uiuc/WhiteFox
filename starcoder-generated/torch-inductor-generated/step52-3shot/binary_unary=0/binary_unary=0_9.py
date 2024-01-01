
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2):
        t1 = self.conv1(x1)
        t2 = t1 + x2
        t3 = torch.relu(t2)
        t4 = self.conv2(t3)
        t5 = t4 + x2
        t6 = torch.relu(t5)
        t7 = t6 + x1
        t8 = torch.nn.ReLU()(t7)
        t9 = self.conv3(t8)
        t10 = t9 + t1
        t11 = torch.relu(t10)
        return t11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
