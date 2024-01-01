
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.mul = torch.mul
        self.add = torch.add
    def forward(self, x1):
        t0 = self.conv(x1)
        x36 = self.relu6(t0)
        x38 = self.sigmoid(x36)
        x44 = self.mul(x38, 6)
        x48 = self.add(x36, 3)
        t13 = self.relu6(x48)
        t15 = self.sigmoid(t13)
        t17 = self.mul(x44, t15)
        return t17
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
