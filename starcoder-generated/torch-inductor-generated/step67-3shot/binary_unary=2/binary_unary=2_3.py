
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 48, 3, stride=2, padding=1)
        self.conv4 = torch.nn.Conv2d(48, 32, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = v4 - 0.7
        v6 = F.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
