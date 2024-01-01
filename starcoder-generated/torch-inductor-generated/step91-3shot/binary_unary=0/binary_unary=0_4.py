
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x, k):
        v1 = self.conv1(x)
        v2 = v1 + x
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 + k
        v6 = torch.relu(v5)
        v7 = self.conv3(v6)
        v8 = v7 + k
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
k = torch.randn(1, 16, 64, 64)
