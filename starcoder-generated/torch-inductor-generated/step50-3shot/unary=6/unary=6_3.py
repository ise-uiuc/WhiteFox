
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(1, 60, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(60, 1, 3, stride=1, padding=0)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        t1 = self.conv(x1)
        y = torch.clamp(t1, 0, 6)
        t2 = self.conv1(y)
        t3 = self.conv2(t2)
        t4 = 3 + t3
        t5 = torch.clamp(t4, 0, 6)
        t6 = t3 * t5
        t7 = t6 / 6
        t8 = self.tanh(t7)
        return t8
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
