
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv(x1)
        v2 = self.conv1(x2)
        v3 = self.conv(v1)
        v4 = self.conv2(x3)
        v5 = v2 + v1
        v6 = torch.relu(v5)
        v7 = v3 + v6 
        v8 = torch.relu(v7)
        v9 = v8 + v4
        v10 = torch.relu(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
