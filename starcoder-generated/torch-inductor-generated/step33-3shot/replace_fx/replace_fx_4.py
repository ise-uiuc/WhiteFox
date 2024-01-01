
class TestModule1(torch.nn.Module):
    def __init__(self, conv=torch.conv2d, weight=None):
        super(TestModule1, self).__init__()
        if (weight is None):
            self.conv = conv
        else:
            self.conv = conv(1, 1, kernel_size=(2, 2), groups=1, bias=True,padding=0, stride=1, dilation=1)
            with torch.no_grad():
                self.conv.weight.copy_(weight)
    def forward(self, x):
        x = self.conv(x) 
        return x
# Inputs to the model
x1 = torch.randn(1, 10, 16, 10)
m = torch.rand(1, 1, 2, 2)
tm = TestModule1(conv=TestModule1.conv, weight=m)
