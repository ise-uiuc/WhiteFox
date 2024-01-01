
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 3, stride=2, padding=1, dilation=2)
    def forward(self, x1, padding=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = v1 + padding
        return v2
# Inputs to the model
x1 = torch.randn(7, 23, 64, 64)
