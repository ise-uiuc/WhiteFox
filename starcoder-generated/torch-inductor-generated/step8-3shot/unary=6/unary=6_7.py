
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU6()
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.relu1(x2)
        x4 = x3 / 6
        return x4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
