
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=None, x2=None):
        v1 = self.conv(x1)
        other = torch.randn(v1.shape) if other == None and x2==None else other
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
