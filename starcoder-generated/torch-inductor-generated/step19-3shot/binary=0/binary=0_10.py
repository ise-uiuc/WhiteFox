
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(12, 16, 8, stride=1, padding=4)
    def forward(self, x1, v1=None):
        if v1 is None:
            v1 = torch.zeros(self.conv.out_channels, x1.shape[1], x1.shape[2] // self.conv.stride[0], x1.shape[3] // self.conv.stride[1])
        v2 = self.conv(x1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 12, 256, 256)
