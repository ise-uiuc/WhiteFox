
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 1, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 9, 1, stride=1, padding=1)
    def forward(self, x1, conv_weight=None, conv_bias=None, other=1, x2=None):
        var1 = self.conv1(x1)
        if not None in (conv_weight, conv_bias):
            var1 = torch.nn.functional.linear(var1, conv_weight, conv_bias)
        var2 = self.conv2(var1)
        if not None in (conv_weight, conv_bias):
            var2 = torch.nn.functional.linear(var2, conv_weight, conv_bias)
        v2 = var2 + other
        return v2
# Inputs to the model
x1 = torch.randn(4, 4, 4)
