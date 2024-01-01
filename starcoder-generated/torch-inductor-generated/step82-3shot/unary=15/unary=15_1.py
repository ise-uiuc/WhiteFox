
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.conv2 = torch.nn.Conv2d(3, 32, 3, stride=2)
        self.conv3 = torch.nn.Conv2d(3, 64, 3, stride=2)
        self.conv4 = torch.nn.Conv2d(64, 128, 3)
        self.conv5 = torch.nn.Conv2d(128, 192, 3, stride=2)
        self.conv6 = torch.nn.Conv2d(192, 256, 3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
