
class ModelReLU(torch.nn.Module):
    def __init__(self):
        super(ModelReLU, self).__init__()
        self.conv_1 = torch.nn.Conv2d(1, 8, 3, stride=1, padding=0, dilation=2, groups=2, bias=True) # depthwise convolution with channel multiplier - 1
    def forward(self, x1):
        x2 = torch.relu(self.conv_1(x1))
        return x2
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
