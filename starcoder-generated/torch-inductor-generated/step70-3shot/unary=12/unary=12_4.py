
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, stride=3, padding=1)
        self.pooling = torch.nn.MaxPool2d(3, stride=3, padding=1)
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pooling(v1)
        v3 = self.conv1(v2)
        v4 = torch.sigmoid(v3)
        v5 = v3 * v4
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
