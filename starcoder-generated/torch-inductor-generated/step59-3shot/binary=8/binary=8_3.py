
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(6, 4, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(4, 3, 3, stride=2, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(9)
        self.bn2 = torch.nn.BatchNorm2d(4)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3.reshape([v3.shape[0], -1])
        v5 = self.bn1(v4)
        v6 = v5.reshape(v3.shape)
        v7 = self.bn2(v6)
        return v7
# Inputs to the model
x = torch.randn(2, 3, 64, 64)
