
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 1, stride=2)
        self.conv2 = torch.nn.Conv2d(5, 7, 1, stride=1)
        self.conv3 = torch.nn.Conv2d(7, 12, 1, stride=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 > 0
        v5 = v3 * -0.015
        v6 = torch.where(v4, v3, v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 220, 220)
