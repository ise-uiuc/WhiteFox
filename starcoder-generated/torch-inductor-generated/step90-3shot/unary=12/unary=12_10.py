
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 24, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(24, 24, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(24, 20, 1, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv2(v2)
        v4 = self.sigmoid(v1)
        v5 = torch.mul(v1, v4)
        v6 = torch.mul(v2, v4)
        v7 = self.sigmoid(v3)
        v8 = torch.mul(v3, v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 25, 25)
