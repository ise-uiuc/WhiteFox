
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(x1)
        v4 = v3 + v1
        v5 = torch.relu(v4)
        v6 = self.conv3(x1)
        a1 = self.conv1(v5)
        a2 = self.conv2(v5)
        a3 = self.conv3(v5)
        v7 = v6 + a1 + a2 + a3
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 255, 73)
