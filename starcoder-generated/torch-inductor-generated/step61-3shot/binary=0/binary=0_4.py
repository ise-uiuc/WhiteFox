
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 3, 1, stride=1, padding=1)
    def forward(self, x1, x2=None):
        v1 = self.conv(x1)
        if x2 == None:
            x2 = torch.randn(v1.shape)
        v2 = v1 + x2
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
