
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(10, 30, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 0.5
        v3 = F.sigmoid(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 1.0
        v6 = F.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
