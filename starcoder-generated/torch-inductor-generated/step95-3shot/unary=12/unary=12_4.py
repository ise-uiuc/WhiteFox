
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 8, 3, stride=1)
        self.conv2 = torch.nn.Conv2d(8, 16, 3, stride=1)
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv(x1))
        v2 = self.conv(x1) * v1
        v3 = torch.sigmoid(self.conv2(v2))
        v4 = self.conv2(v2) * v3
        v5 = torch.sigmoid(torch.Conv2d(4, 16, 3, stride=1)(v4))
        return v5
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
