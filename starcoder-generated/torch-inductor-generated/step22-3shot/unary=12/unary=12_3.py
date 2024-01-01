
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 2, 1, stride=1, padding=0)
        self.conv1 = torch.nn.Conv2d(2, 2, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(2, 2, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(x1)
        v3 = self.conv2(x1)
        v4 = torch.sigmoid(v2)
        v5 = torch.sigmoid(v3)
        v6 = v1 * v4
        v7 = v1 * v5
        return v6, v7
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
