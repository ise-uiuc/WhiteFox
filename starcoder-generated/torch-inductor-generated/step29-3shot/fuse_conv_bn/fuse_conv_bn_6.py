
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(5, 4, 3)
        self.layer2 = torch.nn.BatchNorm2d(4)
    def forward(self, x1):
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        del x3
        x4 = self.layer2(x1)
        return (x2, x4)
# Inputs to the model
x1 = torch.randn(1, 5, 2, 2)
