
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        v2 = other
        if other is None:
            other = torch.randn(v1.shape)
            if v1.shape[0] == 3:
                v2 = torch.randn(v1.shape)
            v3 = v2 + v1
            return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
