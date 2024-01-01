
class Model(torch.nn.Module):
    def __init__(self):
        super().init()
        conv = torch.nn.Conv2d(96, 32, 3, 1, 1)
        bn = torch.nn.BatchNorm2d(32)
        self.layer1 = torch.nn.Sequential(conv, bn)
    def forward(self, x3):
        x3 = self.layer1(x3)
        return x3
# Inputs to the model
x3 = torch.randn(1, 96, 56, 56)
