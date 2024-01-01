
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 25, 1)
        self.conv2 = torch.nn.Conv2d(25, 40, 1)
        self.conv3 = torch.nn.Conv2d(40, 80, 1)
        self.conv4 = torch.nn.Conv2d(80, 160, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv4(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 10, 32, 32)
