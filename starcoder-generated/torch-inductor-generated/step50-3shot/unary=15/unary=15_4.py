
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 512, 3, stride=1, padding=1)
        self.bn = torch.nn.BatchNorm2d(512)  # ReLU was applied before this in TF
        self.conv2 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(512, 1024, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(1024, 256, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = torch.relu(v4)
        v6 = self.conv3(v5)
        v7 = torch.relu(v6)
        v8 = self.conv4(v7)
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x1 = torch.randn(1, 3, 512, 512)
