
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 2, 2, stride=1, padding=1)
    def forward(self, x1, other=None, padding1=0):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other + padding1
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 124, 124)
