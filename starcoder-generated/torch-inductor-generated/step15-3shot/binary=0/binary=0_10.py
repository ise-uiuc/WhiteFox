
model = torch.nn.modules.conv2d._ConvNd(4, 4, [2, 2], [1, 1], [0, 0], 1, 8, False, [0, 0], 1, False, False)
conv = torch.nn.Conv2d(4, 4, 1, stride=1, padding=1)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = conv
    def forward(self, x1, other=None, padding1=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
