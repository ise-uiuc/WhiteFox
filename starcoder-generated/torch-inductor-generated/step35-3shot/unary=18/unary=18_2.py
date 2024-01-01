
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(23, 52, 3, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(52, 17, 4, stride=3, padding=1)
        self.conv3 = torch.nn.Conv2d(17, 11, 3, stride=2, padding=2)
        self.conv4 = torch.nn.Conv2d(11, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 23, 100, 100)
