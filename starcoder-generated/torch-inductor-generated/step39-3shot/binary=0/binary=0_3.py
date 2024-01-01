
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 5, 1, stride=1, padding=1)
    def forward(self, x1, other=0, padding=None):
        v1 = self.conv(x1)
        if padding == None:
            padding = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
