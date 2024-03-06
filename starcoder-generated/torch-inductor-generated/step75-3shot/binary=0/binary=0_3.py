
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, stride=1, padding=(0, 0, 1, 1), dilation=(1, 1))
    def forward(self, x1, padding1=0):
        v1 = self.conv(x1)
        if not padding1 is None:
            v1 += padding1
        v2 = v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)