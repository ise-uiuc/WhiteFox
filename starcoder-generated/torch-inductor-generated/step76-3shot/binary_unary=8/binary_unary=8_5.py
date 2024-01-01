
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv3 = torch.nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.bn3 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv4 = torch.nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv5 = torch.nn.Conv2d(64, 32, 5, stride=1, padding=2)
        self.bn5 = torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv6 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.bn6 = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = self.bn1(v1)
        v2 = torch.relu(v1)
        v2 = self.conv2(v2)
        v2 = self.bn2(v2)
        v3 = torch.relu(v2)
        v3 = self.conv3(v3)
        v3 = self.bn3(v3)
        v4 = torch.relu(v3)
        v4 = self.conv4(v4)
        v4 = self.bn4(v4)
        v5 = torch.relu(v4)
        v5 = self.conv5(v5)
        v5 = self.bn5(v5)
        v6 = torch.relu(v5)
        v6 = self.conv6(v6)
        v6 = self.bn6(v6)
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 32, 100, 100)
