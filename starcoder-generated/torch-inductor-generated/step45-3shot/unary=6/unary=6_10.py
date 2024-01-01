
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = torch.nn.Upsample(3)
    def forward(self, x1):
        v1 = self.upsample(x1)
        v2 = 3 + v1
        v3 = v1 * v2
        v4 = v3 / 6
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 28)
