
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.relu6 = torch.nn.ReLU6()
        self.conv2 = torch.nn.Conv2d(8, 32, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu6(v1 + 3)
        v3 = torch.div(v2, 6)
        v4 = self.conv2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
