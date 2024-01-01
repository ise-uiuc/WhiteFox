
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 1, stride=1)
        self.maxpool1 = torch.nn.MaxPool2d(32, stride=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv1(x2)
        v3 = v1 + v2
        v4 = self.maxpool1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
