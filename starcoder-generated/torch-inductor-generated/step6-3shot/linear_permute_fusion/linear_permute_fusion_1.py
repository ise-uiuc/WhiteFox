
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.nn.functional.conv2d(x1, weight=self.weight, bias=self.bias, dilation=2, padding=2, stride=1, groups=1)
        t2 = t1.permute(3, 2, 0, 1)
        return t2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
