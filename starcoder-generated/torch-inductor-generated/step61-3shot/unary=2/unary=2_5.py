
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mul_ = torch.nn.Parameter(torch.tensor(59.0403, dtype=torch.float32))
        self.add_ = torch.nn.Parameter(torch.tensor(-0.0413, dtype=torch.float32))
        self.conv = torch.nn.Conv2d(3, 64, 1, 1, 0, bias=True)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1.mul(self.mul_)
        v3 = v2 + self.add_
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
