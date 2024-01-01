
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 80, 1, stride=1, padding=2, dilation=1, groups=1)
        self.conv2 = torch.nn.Conv2d(80, 1, 3, stride=1, padding=1, dilation=1, groups=1)
    def forward(self, x1):
        v1 = self.conv2(torch.relu(torch.sigmoid(self.conv1(x1))))
        return v1
# Inputs to the model
x1 = torch.randn(1, 32, 28, 28)
