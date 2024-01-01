
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1, stride=1, padding=1)
    def forward(self, x1, padding=None):
        v1 = self.conv(x1)
        if padding == None:
            padding = torch.randn(v1.shape)
        v2 = v1 + x1
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
