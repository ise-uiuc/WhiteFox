
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(100352, 1024)
        self.fc2 = torch.nn.Linear(1024, 1024)
        self.fc3 = torch.nn.Linear(1024, 256)
    def forward(self, x):
        t1 = self.conv(x)
        r = t1.reshape(1, 100352)
        t2 = self.fc1(r)
        t3 = self.fc2(t2)
        t4 = self.fc3(t3)
        t5 = t4 + 3
        t6 = torch.clamp_max(t5, 6)
        t7 = t6.mul(6)

        return t7
# Inputs (with random weights)
x = torch.randn(1, 3, 256, 256)
