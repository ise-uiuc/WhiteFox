
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, stride=1, padding=1)
    def forward(self, x1):
        b = x1.size(0)
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        v4 = v1 + v2 + v3
        v5 = torch.tanh(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
