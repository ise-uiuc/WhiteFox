
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.padding = 3
        self.groups = 2
        self.stride = 1
        self.bias = True
        self.conv = torch.nn.Conv2d(6, 16, (2, 10), self.stride, self.padding, self.dilation, self.groups, self.bias)
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 6, 638, 350)
