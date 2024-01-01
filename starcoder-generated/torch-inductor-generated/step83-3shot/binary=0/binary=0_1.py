
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 5, 1, stride=1, padding=1)
    def forward(self, x1, x2, padding1=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = v1 + padding1
        v3 = v2 + x2
        v4 = v3 + x1
        return v4
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
x2 = torch.randn(1, 5, 64, 64)
