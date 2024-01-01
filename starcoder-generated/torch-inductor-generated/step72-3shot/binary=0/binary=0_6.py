
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 7, 1, stride=1, padding=1)
    def forward(self, x, padding=None):
        if padding == None:
            padding = 1
        v1 = self.conv(x)
        v2 = v1 + padding
        return v2
# Inputs to the model
x = torch.randn(1, 16, 32, 192)
