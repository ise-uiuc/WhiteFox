
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
    def forward(self, x):
        v = self.conv1(x)
        v1 = v * x
        v2 = self.conv2(v1)
        v3 = v2 + v1
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x = torch.randn(2, 16, 64, 64)
