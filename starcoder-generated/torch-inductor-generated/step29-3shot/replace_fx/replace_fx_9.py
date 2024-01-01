
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        w1 = torch.rand_like(x1, dtype=torch.float)
        v1 = x1 + w1
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
