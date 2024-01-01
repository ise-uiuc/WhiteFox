
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 8, 1, stride=1, padding=1)
    def forward(self, x1, other = None, padding=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        if padding == None or True:
            padding = torch.randn(v1.shape)
            return v2
# Inputs to the model
x1 = torch.randn(1, 6, 48, 64)
