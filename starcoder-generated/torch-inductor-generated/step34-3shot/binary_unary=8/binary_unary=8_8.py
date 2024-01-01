
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, (3, 3))
        self.pool = torch.nn.MaxPool2d((2, 2), (2, 2))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pool(v1)
        v3 = self.conv(x1)
        v4 = self.pool(v3)
        v5 = v2 + v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
