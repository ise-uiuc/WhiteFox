
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(17, 10, 1, stride=1, padding=1, groups=3)
    def forward(self, x1, t1=None):
        v1 = self.conv(x1)
        if t1 == None:
            t1 = torch.randn(v1.shape)
        v2 = v1 + t1
        return v2
# Inputs to the model
x1 = torch.randn(3, 17, 64, 64)
