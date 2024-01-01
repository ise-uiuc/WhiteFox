
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(15, 37, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(37, 69, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(69, 101, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(101, 3, 4, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(3, 21, 3, stride=1, padding=0)
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
x1 = torch.randn(1, 15, 29, 29)
