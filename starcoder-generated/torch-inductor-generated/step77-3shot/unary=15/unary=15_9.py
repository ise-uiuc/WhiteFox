
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 128, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(128, 512, 3, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.relu(v2)
        v4 = self.conv3(v3)
        v5 = self.conv4(v4)
        v6 = torch.relu(v5)
        v7 = self.conv5(v6)
        v8 = self.conv6(v7)
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
