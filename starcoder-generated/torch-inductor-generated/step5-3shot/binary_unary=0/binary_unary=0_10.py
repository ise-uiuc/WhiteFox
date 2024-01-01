
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 2, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, stride=1)
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.pool(v1)
        v3 = v2 + x2
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
x2 = torch.randn(1, 16, 32, 32)
