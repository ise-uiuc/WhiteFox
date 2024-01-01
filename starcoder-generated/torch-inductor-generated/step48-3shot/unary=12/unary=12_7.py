
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv(v2)
        v4 = v1 + v2
        v5 = torch.sigmoid(v3)
        v6 = torch.mul(self.conv3(v4), v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
