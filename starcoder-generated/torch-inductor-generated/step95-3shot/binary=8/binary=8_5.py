
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1, stride=1)
        self.conv2 = torch.nn.Conv2d(3, 64, 3, padding=1, stride=2)
        self.conv3 = torch.nn.Conv2d(3, 64, 3, padding=1, stride=4)
        self.conv4 = torch.nn.Conv2d(3, 64, 3, padding=1, stride=8)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv3(x)
        v4 = self.conv4(x)
        v5 = v1 + v2 + v3 + v4
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
