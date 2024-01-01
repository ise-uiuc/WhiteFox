s
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.where(v1 > 0, v1, v1 * _negative_slope)
        return v2

class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.where(v1 > 0, v1, v1 * negative_slope)
        return v2

class Model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.where(v1 > 0, v1, v1 * -boundaries_slope)
        return v2

class Model4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)

    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.where(v1 > 0, v1, v1 * 3)
        return v2

m1 = Model1()
m2 = Model2()
m3 = Model3()
m4 = Model4()

