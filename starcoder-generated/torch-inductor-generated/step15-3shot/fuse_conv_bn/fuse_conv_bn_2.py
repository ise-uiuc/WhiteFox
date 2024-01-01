
class Model2(torch.nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
    def forward(self, x1, x2, x4):
        x1 = torch.nn.functional.conv2d(input, weight=x2, bias=x2, stride=x4, padding=x4, dilation=x1, groups=x1)
        return x1
# Inputs to the model
input = torch.randn(1, 3, 4, 4)
# (weight=[torch.randn(3, 3, 4, 4), torch.randn(3, 3, 4, 4)], bias=[torch.randn(3), torch.randn(3), torch.randn(3)], stride=[1, 3], padding=[6, 2], dilation=[1, 2], groups=2)
x2 = torch.randn(3, 3, 4, 4)
x4 = (1, 3)
