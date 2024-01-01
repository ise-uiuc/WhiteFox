
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=False)
    def forward(self, x1, other=0, size1=None, size2=None):
        v1 = self.conv(x1)
        if size1 == None:
            size1 = other
        if size2 == None:
            size2 = torch.randn(v1.shape)
        v2 = v1 + 0
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
other = torch.randn(8)
