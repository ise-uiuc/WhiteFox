
class LayerWiseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 3, 3, padding=1)
    def forward(self, x1, x2, x3, x4):
        y1 = self.conv1(x1)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        return y1, y2, y3, y4
# Inputs to the model
x1 = torch.randn(1, 3, 10, 10)
x2 = torch.randn(1, 32, 6, 6)
x3 = torch.randn(1, 16, 4, 4)
x4 = torch.randn(1, 3, 2, 2)
