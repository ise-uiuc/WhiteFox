
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Upsample(size=(), scale_factor=5)
    def forward(self, x1):
        v1 = self.up(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
