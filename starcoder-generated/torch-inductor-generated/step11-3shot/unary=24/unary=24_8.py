
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 6, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 6, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = v1 + v2
        v4 = v3 > 0
        v5 = v3 - 0.1
        v6 = torch.where(v4, v3, v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 6, 256, 256)
