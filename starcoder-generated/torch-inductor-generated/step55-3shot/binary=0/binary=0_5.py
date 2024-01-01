
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=None, padding=None):
        if other == None:
            other = torch.randn(x1.shape)
        v1 = self.conv(x1)
        if padding == None:
            padding = torch.randn(v1.shape)
        v2 = v1 + other
        v3 = v2 + padding
        return v3
# Inputs to the model
x1 = torch.randn(3, 3, 64, 64)
