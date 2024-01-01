
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 256, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(256, 256, 3, stride=2, padding=1)
        self.conv5 = torch.nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(512, 2048, 3, stride=2, padding=1)
        self.conv7 = torch.nn.Conv2d(2048, 4096, 3, stride=2, padding=1)
        self.conv8 = torch.nn.Conv2d(4096, 4096, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = torch.relu(self.conv1(x1))
        v2 = torch.relu(self.conv2(v1))
        v3 = torch.relu(self.conv3(v2))
        v4 = torch.relu(self.conv4(v3))
        v5 = torch.relu(self.conv5(v4))
        v6 = torch.relu(self.conv6(v5))
        v7 = torch.relu(self.conv7(v6))
        v8 = torch.relu(self.conv8(v7))
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 60, 60)
