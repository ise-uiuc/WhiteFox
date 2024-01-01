
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x1, x2, x3):
        v1 = self.conv1(x1)
        v1 = v1 + x2
        v1 = v1 + x3
        v1 = v1 + x2
        v1 = v1 + x1
        v1 = torch.relu(v1)
        v1 = self.conv1(x3)
        v1 = v1 + x1
        v1 = torch.relu(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
x3 = torch.randn(1, 16, 64, 64)
