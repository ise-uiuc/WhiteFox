
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 3, (277, 1557), stride=3, padding=1845)
        self.conv_1 = torch.nn.Conv2d(3, 1, 1, 1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.conv_1(x1)
        x3 = (x2 + 1.6303978221893307e-09) * 0.7071067811865475
        x4 = self.tanh(x3)
        return x4
# Inputs to the model
x = torch.randn(2, 6, 277, 1557)
