
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 3, 1, stride=1, padding=1)
    def forward(self, x1, other=1, padding1=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        x2 = torch.cat([v1, padding1], 1)
        v2 = x2 + other
        return v2

# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
