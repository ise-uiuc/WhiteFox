
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 5, stride=1, padding=2, dilation=1)
        self.conv2 = torch.nn.Conv2d(3, 5, 5, stride=1, padding=2, dilation=2)
        self.conv3 = torch.nn.Conv2d(3, 2, 5, stride=1, padding=4, dilation=4)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(x1)
        v3 = self.conv3(x1)
        return v1 + v2 + v3
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
