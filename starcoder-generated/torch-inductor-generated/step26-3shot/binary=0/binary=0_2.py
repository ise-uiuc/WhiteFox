
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(18, 8, 1, stride=1, padding=0)
    def forward(self, x1, out=None):
        v1 = self.conv(x1)
        if out==None:
            out = torch.ones(v1.shape)
        v2 = v1 + out
        return v2
# Inputs to the model
x1 = torch.randn(1, 18, 64, 64)
