
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 6, 3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(6, 1, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(5, 6, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(6, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        v4 = self.conv3(x1)
        v5 = self.conv4(v4)
        v6 = torch.sigmoid(v5)
        v7 = v3 * v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
