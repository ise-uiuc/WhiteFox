
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(21, 44, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(44, 44, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(44, 44, 3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(44, 44, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v1)
        v4 = self.conv4(v1)
        v5 = v2 + v3 + v4
        v6 = torch.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 21, 64, 64)
