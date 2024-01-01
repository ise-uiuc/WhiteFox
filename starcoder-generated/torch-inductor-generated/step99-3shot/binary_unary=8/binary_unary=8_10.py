
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(1, 32, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(64, 128, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(128, 256, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(256, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = v1 + v6
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
