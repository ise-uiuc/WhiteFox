
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 1, stride=1, padding=1, dilation=1)
        self.conv2 = torch.nn.Conv2d(3, 32, 1, stride=1, padding=1, dilation=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = torch.relu(v1)
        v4 = torch.relu(v2)
        return v3 + v4
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
