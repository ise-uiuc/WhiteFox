
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv2d_3 = torch.nn.Conv2d(3, 3, 5, stride=2, padding=0)
        self.conv2d_6 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=0)
        self.conv2d_8 = torch.nn.Conv2d(3, 3, 7, stride=2, padding=0)
        self.negative_slope = negative_slope
    def forward(self, x1):
        x2 = self.conv2d_3(x1)
        x3 = self.conv2d_6(x2)
        x4 = x3 > 0
        x5 = x3 * self.negative_slope
        x6 = torch.where(x4, x3, x5)
        x7 = self.conv2d_8(x1)
        x8 = torch.neg(x7)
        x9 = torch.tanh(x8)
        return x9
negative_slope = -1.5
# Inputs to the model
x1 = torch.randn(2, 3, 15, 15)
