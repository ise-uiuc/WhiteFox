
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(48, 64, 1, stride=1, padding=1)
    def forward(self, x1, padding1=None, other=True):
        v1 = self.conv(x1)
        if other == True or padding1 == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(2, 48, 64, 64)
