
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(64, 64, 1, stride=1, padding=1)
    def forward(self, x, other1=1, padding=None):
        v1 = self.conv(x)
        if padding == None:
            padding = torch.randn(x.shape)
        v = v1 + other1
        return v
# Inputs to the model
x1 = torch.randn(1, 8, 32, 32, 32)
