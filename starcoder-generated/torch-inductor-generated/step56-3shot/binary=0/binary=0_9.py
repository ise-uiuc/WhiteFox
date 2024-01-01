
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 12, 12, stride=2, padding=7, bias=False, dilation=1)
    def forward(self, x1, padding=None):
        if padding == None:
            padding = torch.randn(self.conv(x1).shape)
        v1 = self.conv(x1)
        v2 = v1 + padding
        return v2
# Inputs to the model
x1 = torch.randn(2, 6, 64, 64)
