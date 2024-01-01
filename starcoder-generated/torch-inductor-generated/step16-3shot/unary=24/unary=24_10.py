
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        v1 = self.conv1(x)
        v1 = self.relu(v1)
        v2 = self.conv2(v1)
        v2 = self.relu(v2)
        v3 = v2 > 0
        v4 = v2 * 0.1
        v5 = torch.where(v3, v2, v4)
        v6 = self.relu(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
