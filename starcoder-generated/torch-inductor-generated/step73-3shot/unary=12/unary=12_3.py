
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 1, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 400, 3, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(400, 1, 1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(1, 4000, 3, stride=2, padding=1)
        self.conv6 = torch.nn.Conv2d(4000, 1, 1, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(1, 1600, 3, stride=2, padding=1)
        self.conv8 = torch.nn.Conv2d(1600, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v2 = v2.tanh()
        v3 = self.conv3(x1)
        v4 = self.conv4(v3)
        v4 = 1/ (2 + v4)
        v3 = F.relu(v3 + v4 * self.conv5(v3))
        v4 = self.conv6(v3)
        v4 = (1/4 + v4)
        v3 = F.relu(v3 + v4 * sself.conv7(v3))
        v4 = self.conv8(v3)
        v5 = torch.sigmoid(v4)
        v6 = (v4 * v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 320, 320)
