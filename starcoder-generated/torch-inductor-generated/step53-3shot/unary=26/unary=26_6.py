
class Conv3DBias(torch.nn.Conv3d):
    def __init__(self, in_channels, bias):
        super(Conv3DBias, self).__init__(in_channels, in_channels, 3, stride=1, padding=1, dilation=1)
        self.input_bias = bias

    def forward(self, input):
        return F.conv3d(input, self.weight, self.bias) + self.input_bias
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = Conv3DBias(2, 1)
    def forward(self, x1):
        x2 = torch.sum(x1, dim=3)
        x3 = self.conv_t(x2)
        return x3
# Inputs to the model
x1 = torch.randn(4, 2, 3)
