
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 2, 3, stride=1)
        self.conv3 = torch.nn.Conv2d(2, 4, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(2, 4, 3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(4, 8, 3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(4, 8, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v1)
        v4 = self.conv3(v2)
        v5 = torch.relu(v4)
        v6 = self.conv4(v4)
        v7 = self.conv5(v5)
        v8 = torch.relu(v7)
        v9 = self.conv6(v7)
        return v8 + v9
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
