
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, dilation=1)
    def forward(self, x1, input_1, bias=None):
        v1 = self.conv(x1)
        if bias == None:
            bias = torch.randn(v1.shape)
        y1 = v1 + input_1
        y1 += bias
        return y1
# Inputs to the model
x1 = torch.randn(1, 1, 8, 8)
x2 = torch.randn(1, 1, 8, 8)
