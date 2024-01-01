
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = self.gelu(x1)
        v2 = v1 * v1 * v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
