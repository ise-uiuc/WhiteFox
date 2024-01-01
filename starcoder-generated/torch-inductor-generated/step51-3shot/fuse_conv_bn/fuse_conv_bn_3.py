
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        x = torch.ops.aten.conv2d(x, weight, bias, stride, padding, dilation, groups)
        return x
# Inputs to the model
x = torch.randn(1024, 35, 35)
w = torch.randn(35, 1024, 5, 5)
