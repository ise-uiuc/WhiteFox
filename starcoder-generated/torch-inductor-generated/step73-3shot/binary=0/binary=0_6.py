
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 5, 3, stride=1, padding=1)
    def forward(self, x1, other=1, padding=None):
        v1 = self.conv(x1)
        if padding == None:
            padding = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(8, 16, 128, 128)
