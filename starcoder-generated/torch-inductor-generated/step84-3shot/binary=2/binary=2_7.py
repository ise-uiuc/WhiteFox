
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, 1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 99.99999999999136
        return v2
# Inputs to the model
x = torch.randn(1, 1, 9, 27)
