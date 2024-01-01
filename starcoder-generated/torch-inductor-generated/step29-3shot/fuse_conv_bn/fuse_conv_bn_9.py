
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(5, 5, 2)
        self.layer2 = torch.nn.BatchNorm2d(5)
        self.layer3 = torch.nn.Conv2d(5, 5, 2)
    def forward(self, x3):
        x1 = self.layer1(x3)
        x4 = self.layer2(x1)
        x2 = self.layer3(x4)
        return (x1, x2, x4)
# Inputs to the model
x3 = torch.randn(1, 5, 4, 4)
