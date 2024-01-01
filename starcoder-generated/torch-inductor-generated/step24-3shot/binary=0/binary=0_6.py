
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.InstanceNorm2d(16, affine=True)
    def forward(self, x1, other=None):
        v1 = self.op(x1)
        if other == None:
            other = torch.ones(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
