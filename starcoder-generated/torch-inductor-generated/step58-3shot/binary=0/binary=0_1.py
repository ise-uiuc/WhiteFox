
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1)
    def forward(self, x1, other=True, other1=False):
        v1 = self.conv(x1)
        if other == True:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        v3 = v2 + other1
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
