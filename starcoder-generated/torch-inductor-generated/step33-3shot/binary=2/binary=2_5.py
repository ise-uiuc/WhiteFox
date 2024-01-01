
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 2048, 1, stride=1, padding=1)
    def forward(self, x2):
        v1 = self.conv1(x2) + 1.1
        v2 = self.conv2(v1) + 2.2
        v3 = self.conv3(v2) + 3.3
        v4 = v3 - 23.4
        return v4
# Inputs to the model
x2 = torch.randn(1, 32, 32, 64)
