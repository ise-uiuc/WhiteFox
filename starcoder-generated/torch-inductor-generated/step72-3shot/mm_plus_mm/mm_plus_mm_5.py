
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(456, 784)
        self.layer2 = nn.Linear(784, 256)
        self.layer3 = nn.Linear(256, 98)
        self.layer4 = nn.Linear(98, 68)
    def forward(self, x1, x2, x3, x4, x5):
        x = self.layer1(x1)
        y = self.layer2(x4)
        z = self.layer3(y)
        w = self.layer4(z)
        return 0.25*((x + w) - (x + y) + (x + z) - x) + x5
# Inputs to the model
x1 = torch.randn(6, 456)
x2 = torch.randn(6, 456)
x3 = torch.randn(6, 456)
x4 = torch.randn(6, 456)
x5 = torch.randn(6, 456)
