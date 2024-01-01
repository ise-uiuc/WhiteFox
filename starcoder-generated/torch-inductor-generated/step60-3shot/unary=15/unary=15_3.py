
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(8, 16, 3, 2, 1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 2, 1)
        self.conv3 = torch.nn.Conv2d(8, 32, 3, 2, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = self.conv3(x1)
        return v4
# Inputs to the model
x1 = torch.randn(2, 8, 256, 256)
