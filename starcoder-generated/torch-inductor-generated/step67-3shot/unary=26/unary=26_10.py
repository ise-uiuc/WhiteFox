
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv2d1 = torch.nn.Conv2d(14, 60, 41, stride=5, padding=3)
        self.conv2d2 = torch.nn.ConvTranspose2d(24, 81, 10, stride=1, padding=4, dilation=6)
        self.conv2d3 = torch.nn.Conv2d(37, 25, 23, stride=4, padding=2)
        self.conv2d4 = torch.nn.Conv2d(13, 41, 7, stride=3, padding=1)
    def forward(self, x7):
        y1 = torch.nn.functional.relu(self.conv2d1(x7))
        y2 = self.conv2d2(y1)
        y3 = self.conv2d3(torch.nn.functional.relu(self.conv2d4(x7)))
        y4 = torch.tanh(torch.add(x7, y2))
        y5 = torch.mul(0.1722, y2)
        y6 = torch.add(torch.neg(x7), y4)
        return y4
# Inputs to the model
x7 = torch.randn(1, 14, 62, 89)
