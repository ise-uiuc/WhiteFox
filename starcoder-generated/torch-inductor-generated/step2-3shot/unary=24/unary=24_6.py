
class Model(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1, bias=True)
        self.param = torch.nn.Parameter(torch.tensor(-1, dtype=dtype))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.param.view(-1, 1, 1, 1)
        v3 = v1 > 0
        v4 = v2 * (-1.01)
        v5 = torch.where(v3, v1, v4)
        return v5
dtype = torch.float
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
