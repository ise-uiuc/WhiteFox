
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 7)
    def forward(self, x1, f=None):
        v1 = self.conv(x1)
        if f == None:
            f = torch.zeros(v1.shape)
        else:
            f = 1
        v2 = v1 + torch.rand(v1.shape)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 10, 10)
f=True
