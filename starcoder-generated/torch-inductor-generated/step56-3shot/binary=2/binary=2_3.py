
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, 1, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(7, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = v2 - v1
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
