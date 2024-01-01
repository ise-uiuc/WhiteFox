
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.rand_like(x1)
        x4 = torch.rand_like(x1)
        x5 = torch.rand_like(x1)
        x2 = x2 * 0.5
        x3 = x3 * 0.5
        x4 = x4 * 0.5
        x5 = x5 * 0.5
        return x1 + x2 + x3 + x4 + x5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
