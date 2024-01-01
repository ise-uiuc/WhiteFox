
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = v1 - 13.0
        v3 = self.conv1(x2)
        v4 = v3 - 14.25
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
