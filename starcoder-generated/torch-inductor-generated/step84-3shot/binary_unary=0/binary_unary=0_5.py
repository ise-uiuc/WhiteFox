
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, 1, stride=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return self.conv3(v1)
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
