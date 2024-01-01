
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = self.conv3(v3)
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
