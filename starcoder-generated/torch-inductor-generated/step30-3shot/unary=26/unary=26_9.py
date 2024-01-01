
# Please note the following line would throw an error:
# torch.nn.functional.conv_transpose2d(1, 2, 2, stride=1)
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 2, stride=1)
        self.negative_slope = negative_slope
    def forward(self, x):
        t1 = self.conv(x)
        t2 = t1 > 0
        t3 = t1 * self.negative_slope
        t4 = torch.where(t2, t1, t3)
        return t4
negative_slope = 2.44
# Inputs to the model
x = torch.randn(10, 1, 3, 3)
