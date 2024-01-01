
class ModelA(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 4, 1, stride=1, padding=1)
        self.param = param

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 1.0
        return v2
class ModelB(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ModelA(1.0)

    def forward(self, x1):
        return self.model(x1)
# Inputs to the model
x1 = torch.randn(1, 5, 28, 28)
