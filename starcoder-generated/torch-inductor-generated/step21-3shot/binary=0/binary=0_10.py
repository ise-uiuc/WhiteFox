
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 6, ((2, 2), (1, 1)), stride=(1, 1), padding=(2, 0))
    def forward(self, x1, other=None, padding1=None, padding2=None):
        v1 = self.conv(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
