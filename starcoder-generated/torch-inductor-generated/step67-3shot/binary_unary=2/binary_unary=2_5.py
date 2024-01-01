
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, 1, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(64, 64, 1, stride=2, padding=3)
        self.conv3 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(64, 64, 1, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 - 0.3
        v5 = F.relu(v4)
        return self.conv4(v5)
# Inputs to the model
x1 = torch.randn(1, 64, 128, 128)
