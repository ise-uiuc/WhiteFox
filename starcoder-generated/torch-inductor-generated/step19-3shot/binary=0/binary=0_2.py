
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=None, padding1=None):
        v1 = self.conv(x1)
        if v1.shape[0] == 3:
            if padding1 is None:
                padding1 = torch.randn(v1.shape)
            elif padding1.shape[0] == 2:
                padding1 = torch.randn(v1.shape)
        v2 = v1 + v1
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
