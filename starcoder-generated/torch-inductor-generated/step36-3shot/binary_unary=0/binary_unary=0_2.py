
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x5, x6):
        v1 = self.conv1(x5)
        v2 = torch.relu(v1)
        v3 = v2 + x6
        v4 = self.conv2(v3)
        return v4
# Inputs to the model
x5 = torch.randn(1, 16, 64, 64)
x6 = torch.randn(1, 16, 64, 64)
