
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 64, 3, groups=6)
    def forward(self, x1, bias=None, padding=None):
        v1 = self.conv(x1)
        if bias == None:
            bias = torch.randn(v1.shape)
        v2 = v1 + bias
        return v2
# Inputs to the model
x1 = torch.randn(1, 12, 32, 32)
