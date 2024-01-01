
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 24, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        return v2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
    def forward(self, x1):
        # v1 = self.conv(x1)
        v2 = Model1()(x1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
