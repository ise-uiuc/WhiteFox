
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, other=None):
        v1 = self.conv(x1)
        v1_tmp = torch.zeros(v1.shape, dtype=torch.bool)
        if other == None:
            other = torch.ones(v1.shape[0], dtype=torch.bool)
        v1_tmp[torch.arange(v1.shape[0]), :, :, :] = other
        v2 = v1 + v1_tmp
        v3 = torch.min(v2, other)
        return v3
# Inputs to the model
x1 = torch.randn(3, 24, 64, 64)
