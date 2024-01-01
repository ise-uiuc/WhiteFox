
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv2d(6, 6, 1, bias=True)
        self.layer2 = torch.nn.BatchNorm2d(6)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x + x
        return x
# Inputs to the model
x = torch.randn(1, 6, 6, 6)
