
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x, y):
        v1 = self.conv1(x)
        v2 = v1 + x
        v3 = torch.relu(v2)
        v4 = v3 + y
        v5 = self.conv2(x)
        v6 = v5 + x
        v7 = self.conv3(v4)
        v8 = v7 + y
        v9 = torch.relu(v8)
        return v9
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
y = torch.randn(1, 16, 64, 64)
