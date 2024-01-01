
class ModelA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        return v1

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = ModelA()
        self.conv = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v = self.model_a(x1)
        v1 = self.conv(v)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
