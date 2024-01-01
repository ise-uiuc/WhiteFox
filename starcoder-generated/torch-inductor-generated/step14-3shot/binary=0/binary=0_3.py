
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(13, 3, 5, stride=1, padding=0)
    def forward(self, x1, other=2, padding1=None, stride1=None):
        v1 = self.conv(x1)
        if stride1 == None:
            stride1 = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(10, 13, 28, 28)
