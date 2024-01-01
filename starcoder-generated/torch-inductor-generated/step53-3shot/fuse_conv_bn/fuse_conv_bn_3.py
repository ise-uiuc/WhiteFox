
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.conv3 = torch.nn.Conv2d(3, 3, 3)
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        return y3
# Inputs to the model
x = torch.randn(1, 3, 10, 10)
