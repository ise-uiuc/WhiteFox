
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 2)
        self.conv2 = torch.nn.Conv2d(4, 16, 2)
        self.conv3 = torch.nn.Conv2d(16, 64, 3)
        self.conv4 = torch.nn.Conv2d(4, 8, 2)
    def forward(self, x):
        v1 = self.conv3(self.conv2(self.conv1(x)))
        t1 = v1 - 1
        t2 = t1 + 48
        t3 = t2 - 6
        t4 = F.relu(t3)
        v2 = self.conv4(x)
        t5 = v2 - 1
        t6 = t5 - 8
        t7 = t6 + 99
        t8 = F.relu(t7)
        return t8
# Inputs to the model
x = torch.randn(2, 1, 48, 48)
