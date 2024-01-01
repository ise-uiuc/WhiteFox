
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(76, 128, 3, stride=3, padding=1)
    def forward(self, x1, bias=None, kernel, padding=None, stride_):
        v1 = self.conv(x1)
        if bias == None:
            bias = torch.randn(v1.shape)
        if padding == None:
            padding = torch.randn(v1.shape)
        if stride_ == None:
            stride_ = torch.randn(v1.shape)
        v2 = v1 + bias
        return v2
# Inputs to the model
x1 = torch.randn(2, 76, 19, 19)
kernel = torch.randn(128, 76, 3, 3)
