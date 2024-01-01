
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(8)
    def forward(self, x):
        negative_slope = 0.00992063
        v1 = self.batch_norm(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(8, 8, 32, 32)
