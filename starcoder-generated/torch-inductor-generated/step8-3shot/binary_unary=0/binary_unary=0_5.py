
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(16, 16, 5, stride=1, padding=2)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.pool(x)
        v3 = self.conv2(v2)
        v4 = v3 + v1
        return torch.relu(v4)
# Inputs to the model
x = torch.randn(1, 16, 64, 64)
