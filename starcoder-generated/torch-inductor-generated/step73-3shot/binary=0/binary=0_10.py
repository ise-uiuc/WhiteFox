
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 16, 1, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        if other == None:
            other = v1 + torch.randn(v1.shape)
        v2 = v1 - other
        return v2 + other
# Inputs to the model
x1 = torch.randn(10, 32, 32, 32)
