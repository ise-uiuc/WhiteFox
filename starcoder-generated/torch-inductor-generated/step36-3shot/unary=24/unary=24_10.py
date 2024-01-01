
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(31, 28, 7, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(28, 21, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 > 0
        v3 = v1 * 44.0
        v4 = torch.where(v2, v1, v3)
        v1 = self.conv2(v4)
        v2 = v1 > 0
        v3 = v1 * 48.5
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 31, 38, 22)
