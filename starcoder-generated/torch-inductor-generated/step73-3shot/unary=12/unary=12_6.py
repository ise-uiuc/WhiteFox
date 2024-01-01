
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 2, 1, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(1, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.pool(v2)
        v4 = self.conv(v3)
        v5 = torch.sigmoid(v4)
        v6 = v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
