
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.conv5 = torch.nn.Conv2d(128, 128, 4, stride=1, padding=1)
        self.gapool1 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 10)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 4
        v4 = F.relu(v3)
        v5 = self.conv3(v4)
        v6 = self.conv4(v5)
        v7 = self.conv5(v6)
        v8 = v7 - 0.5
        v9 = F.relu(v8)
        v10 = self.gapool1(v9)
        v11 = v10.view(v10.size(0), -1)
        v12 = self.fc1(v11)
        x2 = F.log_softmax(v12, dim=1)
        return x2
# Inputs to the model
x1 = torch.randn(1, 3, 362, 362)
