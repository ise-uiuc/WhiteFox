
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=4, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 5, stride=4, padding=2)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = v1 + x1
        v3 = torch.relu(v2)
        v5 = self.conv2(v3)
        v6 = v1 + v5
        v7 = torch.relu(v6)
        v9 = self.conv3(v7)
        v10 = v1 + v9
        v11 = torch.relu(v10)
        return v11
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
