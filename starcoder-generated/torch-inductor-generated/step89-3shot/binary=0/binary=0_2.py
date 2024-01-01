
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 6, 1, stride=1, padding=2)
        self.other_layer = SomeOtherLayer()
    def forward(self, x1, other):
        v1 = self.conv(x1)
        v2 = v1 + other + self.other_layer()
        return v2
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
other = 1
