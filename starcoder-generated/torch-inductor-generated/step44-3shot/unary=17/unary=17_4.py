
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = torch.nn.AdaptiveMaxPool2d(1)
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = self.pooling(x1)
        v2 = self.gelu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 1, 32)
