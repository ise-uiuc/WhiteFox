
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.conv3 = torch.nn.Conv2d(16, 32, 5, stride=2, padding=2)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = self.conv3(v1)
        v4 = v3 - v2
        return v4
# Inputs to the model
x = torch.randn(1, 16, 28, 28)
