
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 16, 1, stride=1, padding=1)
    def forward(self, x1, other=1, other1=0.9, other2=.9, padding1=None, padding2=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = v1 + other
        v3 = v2 + other1
        v4 = v3 + other2
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
