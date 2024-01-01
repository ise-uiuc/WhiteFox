
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 3), torch.nn.ReLU())
        self.layer2 = torch.nn.Sequential(torch.nn.BatchNorm2d(4), torch.nn.Conv2d(4, 4, 3))
        self.layer3 = torch.nn.BatchNorm2d(4)
    def forward(self, x):
        x = self.layer1(x)
        y = self.layer2[1](x)
        z = self.layer3(y + x)
        return z
# Inputs to the model
x = torch.randn(1, 4, 4, 4)
