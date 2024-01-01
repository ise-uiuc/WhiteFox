
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.conv1.weight = torch.nn.Parameter(torch.eye(8, 3).view(8, 3, 5, 5))
        self.conv2 = torch.nn.Conv2d(8, 8, 3, stride=2, padding=1)
        self.conv2.weight = torch.nn.Parameter(torch.eye(8, 8).view(8, 8, 3, 3))
        self.conv3 = torch.nn.Conv2d(8, 8, 3, stride=1, padding=1)
        self.conv3.weight = torch.nn.Parameter(torch.eye(8, 8).view(8, 8, 3, 3))
        self.conv4 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.conv4.weight = torch.nn.Parameter(torch.eye(8, 8).view(8, 8, 1, 1))
        self.conv5 = torch.nn.Conv2d(8, 8, 1, stride=1, padding=0)
        self.conv5.weight = torch.nn.Parameter(torch.eye(8, 8).view(8, 8, 1, 1))
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
x1 = torch.randn(1, 3, 352, 352)
