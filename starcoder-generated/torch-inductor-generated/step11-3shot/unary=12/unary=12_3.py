
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 16, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.sigmoid(v1)
        v3 = v2.repeat(1, 2, 1, 1)
        v4 = self.conv2(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
