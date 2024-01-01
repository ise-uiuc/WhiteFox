
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 33, 3, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(33, 204, 2, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(204, 1027, 1, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(1027, 4, 3, stride=2, padding=0)
        self.conv5 = torch.nn.Conv2d(4, 7, 2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(v4)
        v6 = torch.relu(v5)
        v7 = self.conv4(v6)
        v8 = torch.relu(v7)
        v9 = self.conv5(v8)
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 32, 1068, 1068)
