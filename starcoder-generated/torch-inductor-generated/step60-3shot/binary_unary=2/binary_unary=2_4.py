
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 24, 1)
        self.conv2 = torch.nn.Conv2d(24, 64, 5, stride=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv4 = torch.nn.Conv2d(128, 160, 3, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 - 2.5558
        v4 = F.relu(v3)
        v5 = self.conv3(v4)
        v6 = self.conv4(v5)
        v7 = v6 - -13.5163
        v8 = F.tanh(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
