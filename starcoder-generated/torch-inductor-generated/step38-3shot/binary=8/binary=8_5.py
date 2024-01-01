
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv3x3 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1x1(x)
        v2 = self.conv3x3(x)
        v3 = v1 + v2
        return v3
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
