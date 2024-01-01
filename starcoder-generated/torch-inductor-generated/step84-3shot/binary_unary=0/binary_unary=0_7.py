
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1):
        v1 = self.conv1(torch.relu(x1))
        v2 = self.conv2(v1)
        v3 = v2 + torch.relu(x1)
        v4 = torch.nn.ReLU()(v3)
        v5 = self.conv3(v4)
        v6 = self.conv4(v5)
        v7 = v6 + torch.nn.ReLU()(x1)
        v8 = self.conv3(v7)
        v9 = self.conv4(v6)
        return v8 + torch.nn.ReLU()(v9)
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
