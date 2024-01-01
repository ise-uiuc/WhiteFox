
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=0)
    def forward(self, x1, other=2, padding1=None, stride1=None):
        v1 = self.conv(x1)
        if stride1 == None:
            stride1 = torch.randn(v1.shape)
        v2 = v1 + other
        if padding1 == None:
            padding1 = torch.randn(v2.shape)
        v3 = v2 + padding1
        return v3
# Inputs to the model
x1 = torch.randn(10, 3, 7, 7)
