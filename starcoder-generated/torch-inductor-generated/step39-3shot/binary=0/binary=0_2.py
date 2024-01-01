
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 4, 1, stride=1, padding=1)
    def forward(self, x1, other=None, strides1=None, groups1=-1, padding1=None, dilation1=1, bias_tensor=True):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        if padding1 == None:
            padding1 = [1, 1, 1, 1]
        if strides1 == None:
            strides1 = [2, 2]
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 32, 32)
