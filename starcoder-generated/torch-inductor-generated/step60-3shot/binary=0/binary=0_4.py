
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 6, 1, stride=1, padding=1)
    def forward(self, x1, padding1=None):
        _padding1 = padding1
        if _padding1 == None:
            _padding1 = torch.randn(x1.shape, dtype=x1.dtype, device=x1.device)
        v1 = self.conv(x1) + _padding1
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
