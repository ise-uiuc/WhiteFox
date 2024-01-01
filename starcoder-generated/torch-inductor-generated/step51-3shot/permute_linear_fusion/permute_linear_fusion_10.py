
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Conv2d(2, 2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.nn.functional.conv2d(v1, self.linear.weight, bias=self.linear.bias, stride=self.linear.stride, padding=self.linear.padding, dilation=self.linear.dilation, groups=self.linear.groups)
        return v2.reshape([1, 2, 2, 2])
# Inputs to the model
x1 = torch.randn(1, 2, 2)
