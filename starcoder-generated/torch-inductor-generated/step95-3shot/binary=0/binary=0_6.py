
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 6, 3, stride=1, dilation=1)
    def forward(self, features, padding0=1, padding1=0):
        convolution = torch.nn.functional.conv2d(features, self.conv.weight, self.conv.bias,  self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        return convolution
# Inputs to the model
features = torch.randn(1, 10, 64, 64)
