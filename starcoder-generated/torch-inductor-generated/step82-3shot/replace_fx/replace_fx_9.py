
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = x1 + 1.0
        x2 = torch.rand_like(x1)
        x3 = x2 * x2
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
