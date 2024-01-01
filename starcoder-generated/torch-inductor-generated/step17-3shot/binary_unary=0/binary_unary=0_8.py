
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(8, 8, 7, stride=1, padding=3)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.conv3(x1)
        v6 = self.conv4(v4)
        v7 = v5 + v6
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
x2 = torch.randn(1, 8, 64, 64)
