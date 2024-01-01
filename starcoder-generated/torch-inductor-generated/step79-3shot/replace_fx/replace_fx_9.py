
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = torch.rand_like(x2, dtype=torch.float32)
        t = torch.rand_like(x1)
        return x2 + x3
# Inputs to the model
x1 = torch.randn(1, 1, 1, 16)
