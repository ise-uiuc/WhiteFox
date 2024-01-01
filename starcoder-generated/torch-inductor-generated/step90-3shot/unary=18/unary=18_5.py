
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, (3, 3))
        self.conv2 = torch.nn.Conv2d(3, 32, (5, 5))
        self.conv3 = torch.nn.Conv2d(32, 32, (3, 3))
        self.conv4 = torch.nn.Conv2d(32, 64, (3, 3))
        self.conv5 = torch.nn.Conv2d(64, 3, (3, 3))
    def forward(self, x1):
        v1 = torch.sigmoid(self.conv1(x1))
        v2 = torch.sigmoid(self.conv2(v1))
        v3 = torch.sigmoid(self.conv3(v2))
        v4 = torch.sigmoid(self.conv4(v3))
        v5 = torch.sigmoid(self.conv5(v4))
        v6 = v1 + v5
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
