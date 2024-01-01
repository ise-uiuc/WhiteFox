
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        x2 = x2 + 1
        x3 = x3 + 1
        v2 = x2 + v1
        v3 = torch.relu(v2)
        v4 = x3 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
