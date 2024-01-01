
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3)
        self.conv2 = torch.nn.Conv2d(32, 16, 3)
        self.conv3 = torch.nn.Conv2d(16, 8, 3)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = v3 * 0.001
        v5 = self.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 224, 224)
