
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(v1)
        v4 = self.conv4(v3)
        v5 = self.conv5(v2)
        v6 = self.conv6(v5)
        v7 = v1 + v2 + v3 + v4 + v5 + v6
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
