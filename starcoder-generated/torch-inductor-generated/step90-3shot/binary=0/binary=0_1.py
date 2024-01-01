
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(17, 17, 1, stride=1, padding=1)
    def forward(self, x1, other=None, other1=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        if other1 == None:
            other1 = torch.randn(v1.shape)
        v3 = v2 + other1
        return v3
# Inputs to the model
x1 = torch.randn(1, 17, 32, 32)
