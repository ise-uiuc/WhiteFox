
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=0)
        self.features2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.features3 = torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.features4 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 16, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(16)
        )
    def forward(self, x1):
        v1 = self.features1(x1)
        v2 = self.features2(v1)
        v3 = self.features3(v2)
        v4 = self.features4(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
