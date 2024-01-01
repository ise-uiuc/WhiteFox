
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = v1 + x
        v3 = self.conv2(x)
        v4 = v2 + v3
        v5 = torch.relu(v4)
        v6 = self.conv3(x)
        v7 = v6 + x
        v8 = self.conv4(x)
        v9 = v7 + v8
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
