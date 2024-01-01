
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(x1)
        v6 = torch.cat((v4, v5), dim=1)
        v7 = torch.relu(v6)
        v8 = self.conv4(v7)
        return v8
# Inputs to the model
x1 = torch.randn(4, 32, 224, 224)
