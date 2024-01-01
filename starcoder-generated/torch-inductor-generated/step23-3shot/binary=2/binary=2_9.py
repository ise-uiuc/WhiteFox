
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - 0.0001
        v3 = v2 - 0.01
        v4 = v3 - 0.0000023
        v5 = v4 - 1
        v6 = v5 - 1.234
        v7 = v6 - 0.00001234
        return v7
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
