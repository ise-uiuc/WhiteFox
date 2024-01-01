
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
    def forward(self, x1, other=1, padding1=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = v1 + other
        v3 = torch.cat([v2, padding1])
        return v3
# Inputs to the model
x1 = torch.randn(2, 1, 64, 64)
