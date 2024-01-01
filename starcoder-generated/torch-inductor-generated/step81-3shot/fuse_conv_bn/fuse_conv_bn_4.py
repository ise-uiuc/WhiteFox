
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(113)
        self.m1 = torch.nn.Conv2d(256, 256, 3, groups=3, stride=2, padding=1, bias=True)
        torch.manual_seed(11)
        self.m2 = torch.nn.BatchNorm2d(256)
        torch.manual_seed(19)
        self.m3 = MyModule()
    def forward(self, x5):
        x5 = torch.relu(self.m1(x5))
        x5 = self.m3(x5)
        x5 = self.m2(x5)
        return x5
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(11119)
        self.layer = torch.nn.Linear(256, 128)
    def forward(self, x6):
        return torch.sigmoid(self.layer(x6))
# Inputs to the model
x5 = torch.randn(2, 256, 5, 5)
