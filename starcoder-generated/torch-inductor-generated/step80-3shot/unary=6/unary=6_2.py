
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1*2
        t3 = t2/2
        return t2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 5, stride=3, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = t1*2
        t3 = t2/2
        return t3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
