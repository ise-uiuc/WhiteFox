
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv1 = torch.nn.Conv2d(32, 16, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 8, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv1(v1)
        v3 = torch.tanh(v2)
        v4 = self.conv2(v3)
        v5 = torch.tanh(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
