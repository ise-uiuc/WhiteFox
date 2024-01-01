
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(torch.nn.Conv2d(5, 5, 2), torch.nn.BatchNorm2d(5))
    def forward(self, x4, x5):
        out1 = [self.layer(x4), self.layer(x5)]
        out2 = [self.layer(x4), self.layer(x5)]
        return out1[0], out1[1], out2[0] + out2[1]
# Inputs to the model
x4 = torch.randn(4, 5, 4, 4)
x5 = torch.randn(2, 5, 4, 4)
