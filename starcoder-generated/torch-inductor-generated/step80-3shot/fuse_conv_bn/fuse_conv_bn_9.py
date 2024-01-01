
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 2, stride=2)
        self.conv1 = torch.nn.Conv2d(1, 1, 4, stride=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 4, stride=1)
    def forward(self, x):
        y = self.conv(x)
        y1 = self.conv1(x)
        x1 = self.conv2(y)
        return x1
# Input to the model
x = torch.randn(1, 1, 16, 16)
