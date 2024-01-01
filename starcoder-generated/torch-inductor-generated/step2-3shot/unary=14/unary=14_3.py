
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = torch.nn.ConvTranspose2d(1, 1, 20, stride=10, padding=5)
    def forward(self, x1):
        v1 = self.deconv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv2 = torch.nn.ConvTranspose2d(1, 1, 20, stride=10, padding=9)
    def forward(self, x1):
        v1 = self.deconv2(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = Model1()
        self.m2 = Model2()
    def forward(self, x1):
        v1 = self.m1(x1)
        v2 = self.m2(x1)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
