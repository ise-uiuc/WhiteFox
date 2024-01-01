
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features3 = torch.nn.BatchNorm2d(8)
        self.features2 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.features1 = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.features2(x1)
        v2 = self.features1(v1)
        v3 = self.features3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
