
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 16, 3, stride=1, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = v3 + x
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
