
class A(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 7)
    def forward(self, x):
        x = self.layers(x)
        return x
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 4)
        self.a = A()
    def forward(self, x):
        x = torch.cat((x, x, x), dim=1)
        x = self.layers(x)
        x1 = self.a(x)
        x = torch.stack((x, x, x, x), dim=1)
        x = torch.cat((x, x, x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(3, 2)
