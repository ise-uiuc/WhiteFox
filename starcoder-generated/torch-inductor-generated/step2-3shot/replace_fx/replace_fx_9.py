
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x3 = torch.rand_like(x1, dtype=torch.double)
        return x3 + 0.1
# Inputs to the model
x1 = torch.randn(20)
