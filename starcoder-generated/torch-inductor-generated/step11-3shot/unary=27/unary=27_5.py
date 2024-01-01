
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 13, 1, stride=1, padding=4)
    def forward(self, x1, x2):
        y1 = self.conv(x1)
        y2 = torch.clamp_max(y1, min=21.0)
        return y2
class TestModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Model1()
        self.conv = torch.nn.Conv2d(3, 13, 1, stride=1, padding=4)
    def forward(self, x):
        return self.model1(x, x)
