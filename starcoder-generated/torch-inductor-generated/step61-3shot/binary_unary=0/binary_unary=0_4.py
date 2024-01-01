
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x, x2):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        v3 = v2 + x2
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.relu(v5)
        v7 = v6 + x
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
