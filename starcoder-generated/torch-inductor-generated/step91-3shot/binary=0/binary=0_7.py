
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(32, 64, 3, stride=1, padding=0)
    def forward(self, x1, padding1=False):
        v1 = self.conv(x1)
        if padding1 == True:
            padding1 = torch.randn(v1.shape)
        v2 = v1 + 0
        return v2
# Inputs to the model
x1 = torch.randn(1, 32, 64, 64)
x2 = torch.randn(x1.shape)
