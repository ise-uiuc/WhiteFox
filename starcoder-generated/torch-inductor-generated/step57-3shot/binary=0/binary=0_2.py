
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(13, 21, 1)
        self.bn = torch.nn.BatchNorm2d(21)
    def forward(self, x, weight=None, bias=None, eps=None):
        v1 = self.conv(x)
        if weight == None:
            weight = torch.randn(v1.shape)
        if bias == None:
            bias = torch.randn(v1.shape)
        if eps == None:
            eps = torch.randn(v1.shape)
        v2 = self.bn(v1, weight=weight, bias=bias, eps=eps)
        return v2
# Inputs to the model
x = torch.randn(1, 13, 64, 64)
