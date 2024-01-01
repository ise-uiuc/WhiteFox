
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 7, stride=1, padding=1)
    def forward(self, x1, other=None, stride1=1, stride2=1, padding1=1, padding2=1):
        v1 = self.conv(x1)
        if other == None:
            other = torch.ones(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 124, 124)
