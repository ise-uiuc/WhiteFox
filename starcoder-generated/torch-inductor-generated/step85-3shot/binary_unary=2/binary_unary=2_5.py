
class Layer0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(3, 3, 2, stride=1, padding=0)
        self.conv_2 = torch.nn.Conv2d(3, 5, 3, stride=1, padding=0)
        self.conv_3 = torch.nn.Conv2d(5, 7, 1, stride=1, padding=0)
    def forward(self, x0):
        x = self.conv_1(x0)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = F.relu(x)
        return x
class Layer1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_0 = Layer0()
        self.conv_0 = torch.nn.Conv2d(3, 4, 4, stride=1, padding=0)
    def forward(self, x1):
        y = self.layer_0(x1)
        y = self.conv_0(y)
        y = F.relu(y)
        return y
class Layer2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_0 = Layer0()
        self.layer_1 = Layer1()
        self.conv_0 = torch.nn.Conv2d(4, 1, 1, stride=1, padding=0)
    def forward(self, x2):
        z = self.layer_0(x2)
        z = self.layer_1(z)
        z = self.conv_0(z)
        z = F.relu(z)
        return z
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_0 = Layer0()
        self.layer_1 = Layer1()
        self.layer_2 = Layer2()
    def forward(self, x3):
        z = self.layer_0(x3)
        z = self.layer_1(z)
        z = self.layer_2(z)
        return z
# Inputs to the model
x3 = torch.randn(1, 3, 3, 3)
