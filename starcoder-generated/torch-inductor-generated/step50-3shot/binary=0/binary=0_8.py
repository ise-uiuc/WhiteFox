
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(13, 16, 1, stride=2, padding=1)
    def forward(self, x1, padding1=1, t=1):
        v1 = self.conv(x1)
        if t == 1:
            t = torch.randn(v1.shape)
        else:
            padding1 = 1
        v2 = v1 + t
        return v2
# Inputs to the model
x1 = torch.randn(6, 13, 64, 64)
